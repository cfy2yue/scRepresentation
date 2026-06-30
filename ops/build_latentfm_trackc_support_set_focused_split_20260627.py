#!/usr/bin/env python3
"""Build a safe trainselect focused split for Track C support-set training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BASE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
FOOTPRINT_JSON = ROOT / "reports/latentfm_trackc_support_set_footprint_prevalence_gate_20260627.json"
OUT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect_supportset_min2_focused.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_support_set_focused_split_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SET_FOCUSED_SPLIT_20260627.md"


def subset_dataset(base: dict[str, Any], train_rows: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    train_set = set(train_rows)
    for key, value in base.items():
        if isinstance(value, list):
            if key == "train":
                out[key] = list(train_rows)
            elif key == "train_multi":
                out[key] = [str(x) for x in value if str(x) in train_set]
            elif key == "train_single":
                out[key] = []
            elif key.startswith("heldout_query") or key.startswith("canonical_"):
                out[key] = list(value)
            else:
                out[key] = list(value)
        else:
            out[key] = value
    out["usage_rule"] = (
        str(out.get("usage_rule") or "")
        + "; focused_support_set_min2_train_only_token_present_rows_no_heldout_query_selection"
    ).strip("; ")
    return out


def main() -> int:
    reasons: list[str] = []
    base = json.loads(BASE_SPLIT.read_text(encoding="utf-8"))
    footprint = json.loads(FOOTPRINT_JSON.read_text(encoding="utf-8"))
    if footprint.get("status") != "trackc_support_set_footprint_prevalence_gate_pass_focused_split_next_no_gpu":
        reasons.append(f"footprint_gate_not_pass:{footprint.get('status')}")
    rows = footprint.get("counts", {}).get("present_rows") or []
    by_dataset: dict[str, list[str]] = {}
    for row in rows:
        if not row.get("is_train_multi"):
            reasons.append(f"present_row_not_train_multi:{row}")
            continue
        by_dataset.setdefault(str(row["dataset"]), []).append(str(row["condition"]))
    focused: dict[str, Any] = {}
    for ds_name, train_rows in sorted(by_dataset.items()):
        if ds_name not in base:
            reasons.append(f"dataset_missing_in_base:{ds_name}")
            continue
        train_rows = sorted(set(train_rows), key=train_rows.index)
        focused[ds_name] = subset_dataset(base[ds_name], train_rows)
    total_train = sum(len(v.get("train", [])) for v in focused.values())
    total_support_val = sum(len(v.get("support_val_multi", v.get("test_multi", []))) for v in focused.values())
    if total_train != 32:
        reasons.append(f"focused_train_count_expected_32_got_{total_train}")
    if total_support_val != 24:
        reasons.append(f"support_val_count_expected_24_got_{total_support_val}")
    for forbidden_group in ["heldout_query_multi_final_only", "canonical_test_single_final_only"]:
        # These final-only lists are retained as provenance only, not used by the
        # launcher/gate. The focused split still carries them to avoid pretending
        # the canonical split was recut.
        pass
    payload = {
        "status": "trackc_support_set_focused_split_ready_no_gpu" if not reasons else "trackc_support_set_focused_split_fail_no_gpu",
        "gpu_authorized": False,
        "reasons": reasons,
        "base_split": str(BASE_SPLIT),
        "focused_split": str(OUT_SPLIT),
        "source_footprint_gate": str(FOOTPRINT_JSON),
        "counts": {
            "datasets": sorted(focused),
            "train": total_train,
            "support_val_multi": total_support_val,
            "by_dataset": {
                ds: {
                    "train": len(groups.get("train", [])),
                    "train_multi": len(groups.get("train_multi", [])),
                    "support_val_multi": len(groups.get("support_val_multi", groups.get("test_multi", []))),
                    "test_multi": len(groups.get("test_multi", [])),
                }
                for ds, groups in sorted(focused.items())
            },
        },
        "boundary": {
            "derived_from_safe_trainselect": True,
            "heldout_trackc_query_used_for_training_or_selection": False,
            "canonical_multi_selection_used": False,
            "training_or_inference_used": False,
        },
    }
    if not reasons:
        OUT_SPLIT.write_text(json.dumps(focused, indent=2, sort_keys=True), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Track C Support-Set Focused Split 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Counts",
        "",
        f"- train: `{total_train}`",
        f"- support_val_multi: `{total_support_val}`",
        f"- by dataset: `{payload['counts']['by_dataset']}`",
        "",
        "## Boundary",
        "",
        "- Derived only from safe trainselect token-present train rows.",
        "- Held-out Track C query and canonical multi are not used for training or selection.",
        "- This file authorizes only a launcher preflight, not model promotion.",
        "",
        "## Reasons",
        "",
    ]
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines.extend(["", "## Outputs", "", f"- split: `{OUT_SPLIT}`", f"- json: `{OUT_JSON}`", ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "reasons": reasons, "split": str(OUT_SPLIT)}, indent=2))
    return 0 if not reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())
