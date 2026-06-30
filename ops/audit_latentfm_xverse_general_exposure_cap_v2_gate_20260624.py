#!/usr/bin/env python3
"""General dataset/cell exposure cap v2 CPU gate for xverse Track A.

This extends the already-passed Jiang exposure-capped split into a broader
microstep-exposure cap across all non-preserved datasets. It is a split-only
gate: no model outputs, canonical outcomes, Track C query, active logs, or
posthoc predictions are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OPS = ROOT / "ops"
import sys

if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import audit_latentfm_xverse_jiang_exposure_capped_split_gate_20260624 as jiang


BIFLOW = ROOT / "dataset/biFlow_data"
SPLIT_DIR = BIFLOW / "xverse_scaling_splits_v2_20260624"
BASE_SPLIT = SPLIT_DIR / "split_seed42_xverse_trainonly_scaling_jiang_exposure_capped_v2.json"
OUT_SPLIT = SPLIT_DIR / "split_seed42_xverse_trainonly_scaling_general_exposure_cap_v2.json"
OUT_JSON = ROOT / "reports/latentfm_xverse_general_exposure_cap_v2_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_GENERAL_EXPOSURE_CAP_V2_GATE_20260624.md"

PRESERVE_DATASETS = {"NormanWeissman2019_filtered", "Wessels"}
TARGET_MAX_SHARE = 0.085
TARGET_JIANG_SHARE = 0.40
TARGET_ENTROPY = 0.918
MIN_TRAIN_CONDITIONS = 1000
MIN_DATASETS_WITH_TRAIN = 22


def stable_key(seed: int, ds: str, cond: str) -> str:
    return hashlib.sha256(f"{seed}\t{ds}\t{cond}".encode("utf-8")).hexdigest()


def clone_with_removed(split: dict[str, Any], removed: set[tuple[str, str]]) -> dict[str, Any]:
    out = {}
    for ds, groups in sorted(split.items()):
        new = {}
        for key, val in groups.items():
            if isinstance(val, list):
                values = [str(x) for x in val]
                if key == "train":
                    values = [c for c in values if (str(ds), c) not in removed]
                new[key] = values
            else:
                new[key] = val
        out[str(ds)] = new
    return out


def train_condition_count(split: dict[str, Any]) -> int:
    return sum(len(groups.get("train") or []) for groups in split.values())


def datasets_with_train(split: dict[str, Any]) -> int:
    return sum(1 for groups in split.values() if groups.get("train"))


def build_candidate(base: dict[str, Any], *, seed: int) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    sizes = {ds: jiang.condition_sizes(str(ds)) for ds in base}
    removed: set[tuple[str, str]] = set()
    current = clone_with_removed(base, removed)
    for _ in range(500):
        exp = jiang.split_exposure(current)
        if (
            exp["max_dataset_epoch_step_share"] <= TARGET_MAX_SHARE
            and exp["jiang_epoch_step_share"] <= TARGET_JIANG_SHARE
            and exp["normalized_dataset_step_entropy"] >= TARGET_ENTROPY
        ):
            return current, removed
        rows = [r for r in exp["rows"] if r["dataset"] not in PRESERVE_DATASETS and r["train_conditions"] > 5]
        if not rows:
            return current, removed
        if exp["jiang_epoch_step_share"] > TARGET_JIANG_SHARE:
            rows = [r for r in rows if r["dataset"] in jiang.JIANG_DATASETS] or rows
        else:
            rows = [r for r in rows if r["epoch_step_share"] > TARGET_MAX_SHARE] or rows
        target_ds = max(rows, key=lambda r: r["epoch_step_share"])["dataset"]
        train = [str(c) for c in (current.get(target_ds) or {}).get("train") or []]
        if len(train) <= 5:
            return current, removed
        # Remove the largest remaining condition from the highest-share dataset.
        cond = max(train, key=lambda c: (sizes[target_ds].get(c, 0), stable_key(seed, target_ds, c)))
        removed.add((target_ds, cond))
        current = clone_with_removed(base, removed)
    return current, removed


def split_summary(split: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    exp = jiang.split_exposure(split)
    return {
        "train_conditions": train_condition_count(split),
        "datasets_with_train": datasets_with_train(split),
        "perturbation_type_counts": jiang.ptype_counts(split, metadata),
        "exposure": {k: v for k, v in exp.items() if k != "rows"},
        "top_exposure_rows": sorted(exp["rows"], key=lambda r: r["epoch_step_share"], reverse=True)[:12],
    }


def decide(base_summary: dict[str, Any], candidate_summary: dict[str, Any], bg_rows: list[dict[str, Any]], preserve_missing: list[str]) -> list[str]:
    reasons = []
    exp = candidate_summary["exposure"]
    if candidate_summary["train_conditions"] < MIN_TRAIN_CONDITIONS:
        reasons.append("train_conditions_lt_1000")
    if candidate_summary["datasets_with_train"] < MIN_DATASETS_WITH_TRAIN:
        reasons.append("datasets_with_train_lt_22")
    if preserve_missing:
        reasons.append("preserved_dataset_conditions_removed")
    if exp["jiang_epoch_step_share"] > TARGET_JIANG_SHARE:
        reasons.append("jiang_epoch_step_share_gt_0p38")
    if exp["max_dataset_epoch_step_share"] > TARGET_MAX_SHARE:
        reasons.append("max_dataset_epoch_step_share_gt_0p085")
    if exp["normalized_dataset_step_entropy"] < TARGET_ENTROPY:
        reasons.append("normalized_dataset_step_entropy_lt_0p918")
    if exp["normalized_dataset_step_entropy"] <= base_summary["exposure"]["normalized_dataset_step_entropy"] + 0.01:
        reasons.append("entropy_not_materially_above_jiang_base")
    if exp["max_dataset_epoch_step_share"] >= base_summary["exposure"]["max_dataset_epoch_step_share"] - 0.01:
        reasons.append("max_share_not_materially_below_jiang_base")
    for row in bg_rows:
        if row["train_eval_js_divergence"] > 0.10:
            reasons.append(f"{row['dataset']}:train_eval_js_gt_0p10")
        if row["train_max_background_share"] > 0.30:
            reasons.append(f"{row['dataset']}:train_max_bg_share_gt_0p30")
    ptypes = candidate_summary["perturbation_type_counts"]
    if ptypes.get("drug", 0) < 220:
        reasons.append("drug_conditions_lt_220")
    if ptypes.get("CRISPRi", 0) < 450:
        reasons.append("crispri_conditions_lt_450")
    if ptypes.get("CRISPRa", 0) < 120:
        reasons.append("crispra_conditions_lt_120")
    if ptypes.get("CRISPRko", 0) < 120:
        reasons.append("crisprko_conditions_lt_120")
    if ptypes.get("Cas13", 0) < 7:
        reasons.append("cas13_conditions_lt_7")
    return reasons


def render_md(payload: dict[str, Any]) -> str:
    def f(value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    lines = [
        "# LatentFM xverse General Exposure-Cap v2 Gate",
        "",
        f"Status: `{payload['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- CPU-only split-composition gate starting from the Jiang exposure-capped split.",
        "- Reads split files, condition metadata, h5 GT offsets, and Jiang `.obs` perturbation/cell_type for background checks.",
        "- Does not read expression matrices, canonical outcomes, Track C query, model outputs, active logs, or posthoc predictions.",
        "",
        "## Gate Rule",
        "",
        f"- train conditions `>= {MIN_TRAIN_CONDITIONS}` and datasets with train `>= {MIN_DATASETS_WITH_TRAIN}`;",
        f"- Jiang epoch-step share `<= {TARGET_JIANG_SHARE}`;",
        f"- max dataset epoch-step share `<= {TARGET_MAX_SHARE}`;",
        f"- normalized dataset-step entropy `>= {TARGET_ENTROPY}`;",
        "- material improvement over Jiang base: entropy `+0.01`, max-share `-0.01`;",
        "- no Jiang train/eval background JS `> 0.10` and no train background share `> 0.30`;",
        "- perturbation-type floors: drug 220, CRISPRi 450, CRISPRa 120, CRISPRko 120, Cas13 7.",
        "",
        "## Summary",
        "",
        "| arm | train conds | Jiang step share | max dataset step share | entropy | ptypes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in ("base_jiang", "candidate"):
        row = payload[name]
        ptypes = ", ".join(f"{k}:{v}" for k, v in row["perturbation_type_counts"].items())
        lines.append(
            f"| `{name}` | {row['train_conditions']} | {f(row['exposure']['jiang_epoch_step_share'])} | "
            f"{f(row['exposure']['max_dataset_epoch_step_share'])} | "
            f"{f(row['exposure']['normalized_dataset_step_entropy'])} | {ptypes} |"
        )
    lines += [
        "",
        "## Candidate Top Exposure Rows",
        "",
        "| dataset | train conds | epoch step share |",
        "|---|---:|---:|",
    ]
    for row in payload["candidate"]["top_exposure_rows"]:
        lines.append(f"| `{row['dataset']}` | {row['train_conditions']} | {f(row['epoch_step_share'])} |")
    lines += [
        "",
        "## Jiang Background Checks",
        "",
        "| dataset | train conds | train/eval JS | train max background share |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["candidate"]["jiang_background_rows"]:
        lines.append(
            f"| `{row['dataset']}` | {row['train_conditions']} | "
            f"{f(row['train_eval_js_divergence'])} | {f(row['train_max_background_share'])} |"
        )
    lines += ["", "## Removed Conditions", ""]
    for ds, conds in payload["removed_conditions_by_dataset"].items():
        if conds:
            preview = ", ".join(f"`{c}`" for c in conds[:12])
            suffix = " ..." if len(conds) > 12 else ""
            lines.append(f"- `{ds}`: {len(conds)} removed; {preview}{suffix}")
    lines += ["", "## Gate Reasons", ""]
    lines.extend([f"- `{reason}`" for reason in payload["reasons"]] or ["- none"])
    lines += [
        "",
        "## Candidate Split",
        "",
        f"`{payload['files']['candidate_split'] or 'not written because gate failed'}`",
        "",
        "## JSON",
        "",
        f"`{payload['out_json']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-split", type=Path, default=OUT_SPLIT)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    base = jiang.read_json(BASE_SPLIT)
    metadata = jiang.read_json(jiang.DATA_DIR / "condition_metadata.json")
    candidate, removed = build_candidate(base, seed=int(args.seed))
    base_summary = split_summary(base, metadata)
    candidate_summary = split_summary(candidate, metadata)
    bg_rows = jiang.jiang_bg_audit(candidate)
    candidate_summary["jiang_background_rows"] = bg_rows

    preserve_missing = []
    for ds in sorted(PRESERVE_DATASETS):
        base_train = set(str(c) for c in (base.get(ds) or {}).get("train") or [])
        cand_train = set(str(c) for c in (candidate.get(ds) or {}).get("train") or [])
        preserve_missing.extend(f"{ds}:{c}" for c in sorted(base_train - cand_train))

    reasons = decide(base_summary, candidate_summary, bg_rows, preserve_missing)
    status = "general_exposure_cap_v2_gate_pass_no_gpu" if not reasons else "general_exposure_cap_v2_gate_fail_no_gpu"

    if status == "general_exposure_cap_v2_gate_pass_no_gpu":
        args.out_split.parent.mkdir(parents=True, exist_ok=True)
        args.out_split.write_text(json.dumps(candidate, indent=2, ensure_ascii=False), encoding="utf-8")

    removed_by_dataset: dict[str, list[str]] = {}
    for ds, cond in sorted(removed):
        removed_by_dataset.setdefault(ds, []).append(cond)

    payload = {
        "status": status,
        "reasons": reasons,
        "boundary": {
            "uses": ["Jiang exposure-capped train-only split", "condition_metadata", "h5 GT offsets", "Jiang h5ad obs perturbation/cell_type"],
            "forbidden": ["expression matrices", "canonical outcomes", "Track C query", "model outputs", "active logs", "posthoc predictions"],
        },
        "target_thresholds": {
            "max_dataset_epoch_step_share": TARGET_MAX_SHARE,
            "jiang_epoch_step_share": TARGET_JIANG_SHARE,
            "normalized_dataset_step_entropy": TARGET_ENTROPY,
        },
        "files": {"base_split": str(BASE_SPLIT), "candidate_split": str(args.out_split) if status == "general_exposure_cap_v2_gate_pass_no_gpu" else ""},
        "base_jiang": base_summary,
        "candidate": candidate_summary,
        "removed_conditions_by_dataset": removed_by_dataset,
        "preserve_missing": preserve_missing[:30],
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(args.out_md), "candidate_split": payload["files"]["candidate_split"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
