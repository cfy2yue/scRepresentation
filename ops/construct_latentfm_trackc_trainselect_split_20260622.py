#!/usr/bin/env python3
"""Build a Track C training-selection split without query leakage.

`split_seed42_multi_support_v2.json` is a full Track C artifact: its `test`
field contains final query multi conditions for downstream evaluation. That is
not safe to pass directly to `model.latent.train`, because training uses the
split `test` field for epoch-end best-checkpoint selection.

This script creates a companion split for GPU fine-tuning where:

- `train` keeps canonical train singles plus Track C train_multi;
- `test` contains support_val_multi only;
- final query conditions are stored under `heldout_query_multi_final_only`;
- canonical test_single/query_multi are not exposed through `test`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_IN_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
DEFAULT_OUT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_trackc_trainselect_split_audit_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_TRAINSELECT_SPLIT_AUDIT_20260622.md"

FOCUS_DATASETS = ("NormanWeissman2019_filtered", "Wessels")
STRATUM_MAP = {
    "test_multi_seen": "query_multi_seen",
    "test_multi_unseen1": "query_multi_unseen1",
    "test_multi_unseen2": "query_multi_unseen2",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha_list(values: list[str]) -> str:
    joined = "\n".join(sorted(map(str, values)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def construct(source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    out: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    for ds, obj in sorted(source.items()):
        ds = str(ds)
        train = list(map(str, obj.get("train") or []))
        train_multi = list(map(str, obj.get("train_multi") or []))
        support_val_multi = list(map(str, obj.get("support_val_multi") or []))
        query_multi = list(map(str, obj.get("query_multi") or []))
        test_single = list(map(str, obj.get("test_single") or []))
        support_by_stratum = obj.get("support_val_multi_by_stratum") or {}

        test = support_val_multi if ds in FOCUS_DATASETS else []
        train_set = set(train)
        test_set = set(test)
        query_set = set(query_multi)

        train_test_overlap = sorted(train_set & test_set)
        test_query_overlap = sorted(test_set & query_set)
        if train_test_overlap:
            reasons.append(f"{ds}:train_test_overlap={len(train_test_overlap)}")
        if test_query_overlap:
            reasons.append(f"{ds}:test_query_overlap={len(test_query_overlap)}")
        if ds in FOCUS_DATASETS and len(test) == 0:
            reasons.append(f"{ds}:missing_support_val_test")

        out_obj = {
            "train": train,
            "train_single": list(map(str, obj.get("train_single") or [])),
            "train_multi": train_multi,
            "support_val_multi": support_val_multi,
            "test": test,
            "test_single": [],
            "test_multi": test,
            "test_multi_seen": list(map(str, support_by_stratum.get("test_multi_seen") or [])),
            "test_multi_unseen1": list(map(str, support_by_stratum.get("test_multi_unseen1") or [])),
            "test_multi_unseen2": list(map(str, support_by_stratum.get("test_multi_unseen2") or [])),
            "heldout_query_multi_final_only": query_multi,
            "heldout_query_multi_seen_final_only": list(map(str, obj.get("query_multi_seen") or [])),
            "heldout_query_multi_unseen1_final_only": list(map(str, obj.get("query_multi_unseen1") or [])),
            "heldout_query_multi_unseen2_final_only": list(map(str, obj.get("query_multi_unseen2") or [])),
            "canonical_test_single_final_only": test_single,
            "source_query_multi_sha256": sha_list(query_multi),
            "source_canonical_test_single_sha256": sha_list(test_single),
            "usage_rule": (
                "training selection split only; do not use for final query evaluation; "
                "use split_seed42_multi_support_v2.json for one-shot heldout query"
            ),
        }
        out[ds] = out_obj
        rows.append(
            {
                "dataset": ds,
                "role": "focus_support_selection" if ds in FOCUS_DATASETS else "train_only_no_selection_test",
                "train": len(train),
                "train_multi": len(train_multi),
                "selection_test": len(test),
                "heldout_query_final_only": len(query_multi),
                "canonical_test_single_final_only": len(test_single),
                "train_test_overlap": len(train_test_overlap),
                "test_query_overlap": len(test_query_overlap),
            }
        )
    payload = {
        "status": "pass_no_query_in_training_test" if not reasons else "fail",
        "reasons": reasons,
        "rows": rows,
    }
    return out, payload


def render(payload: dict[str, Any], *, in_split: Path, out_split: Path, split_sha256: str) -> str:
    lines = [
        "# LatentFM Track C Train-Selection Split Audit",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Provenance",
        "",
        f"- source_trackc_split: `{in_split}`",
        f"- train_selection_split: `{out_split}`",
        f"- train_selection_split_sha256: `{split_sha256}`",
        "- leakage_status: `test_field_contains_support_val_multi_only_no_query_multi_no_canonical_test_single`",
        "",
        "## Why This Exists",
        "",
        "`model.latent.train` uses the split `test` field for epoch-end best-checkpoint selection. "
        "The full Track C v2 split keeps final query multi in `test`, so it must not be used directly for training.",
        "",
        "## Counts",
        "",
        "| dataset | role | train | train multi | selection test | heldout query final only | canonical test single final only | train/test overlap | test/query overlap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['role']} | {row['train']} | {row['train_multi']} | "
            f"{row['selection_test']} | {row['heldout_query_final_only']} | "
            f"{row['canonical_test_single_final_only']} | {row['train_test_overlap']} | "
            f"{row['test_query_overlap']} |"
        )
    lines += ["", "## Decision Reasons", ""]
    reasons = payload.get("reasons") or []
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Usage Rule",
        "",
        "- Use this split only for Track C adapter/router training-time checkpoint selection.",
        "- Do not use this split for final held-out query reporting.",
        "- Final query evaluation must use `split_seed42_multi_support_v2.json` once, after all choices are frozen.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-split", type=Path, default=DEFAULT_IN_SPLIT)
    parser.add_argument("--out-split", type=Path, default=DEFAULT_OUT_SPLIT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    source = load_json(args.in_split)
    split, payload = construct(source)
    args.out_split.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_split.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    split_sha256 = hashlib.sha256(args.out_split.read_bytes()).hexdigest()
    payload.update(
        {
            "source_trackc_split": str(args.in_split),
            "train_selection_split": str(args.out_split),
            "train_selection_split_sha256": split_sha256,
            "leakage_status": "test_field_contains_support_val_multi_only_no_query_multi_no_canonical_test_single",
        }
    )
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render(payload, in_split=args.in_split, out_split=args.out_split, split_sha256=split_sha256), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_split": str(args.out_split), "out_md": str(args.out_md)}, indent=2))
    return 0 if payload["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
