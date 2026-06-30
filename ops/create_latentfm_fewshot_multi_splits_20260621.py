#!/usr/bin/env python3
"""Create no-leakage few-shot multi-condition LatentFM split variants.

The canonical split keeps all multi-gene conditions in test.  These variants
move a small deterministic subset of multi conditions into train to test
whether held-out multi-unseen2 performance is an exposure/identifiability
problem.  Split JSONs intentionally contain only dataset keys so they can be
passed directly to ``model.latent.train --split-file``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_ARMS: dict[str, dict[str, dict[str, int]]] = {
    "norman_multi16": {
        "NormanWeissman2019_filtered": {
            "test_multi_seen": 8,
            "test_multi_unseen1": 8,
        },
    },
    "wessels_multi16": {
        "Wessels": {
            "test_multi_seen": 8,
            "test_multi_unseen1": 8,
        },
    },
    "norman_wessels_multi32": {
        "NormanWeissman2019_filtered": {
            "test_multi_seen": 8,
            "test_multi_unseen1": 8,
        },
        "Wessels": {
            "test_multi_seen": 8,
            "test_multi_unseen1": 8,
        },
    },
    "norman_wessels_gasperini_multi33": {
        "NormanWeissman2019_filtered": {
            "test_multi_seen": 8,
            "test_multi_unseen1": 8,
        },
        "Wessels": {
            "test_multi_seen": 8,
            "test_multi_unseen1": 8,
        },
        "GasperiniShendure2019_lowMOI": {
            "test_multi_unseen2": 1,
        },
    },
}

GROUPS_TO_REMOVE = (
    "test",
    "test_multi",
    "test_multi_seen",
    "test_multi_unseen1",
    "test_multi_unseen2",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_pick(items: list[str], *, n: int, seed: str, arm: str, dataset: str, group: str) -> list[str]:
    keyed = []
    for item in items:
        digest = hashlib.sha256(f"{seed}|{arm}|{dataset}|{group}|{item}".encode("utf-8")).hexdigest()
        keyed.append((digest, item))
    return [item for _, item in sorted(keyed)[: min(n, len(keyed))]]


def is_multi(condition: str) -> bool:
    return "+" in str(condition)


def split_counts(split: dict[str, Any]) -> dict[str, Any]:
    totals = {
        "datasets": 0,
        "train": 0,
        "test": 0,
        "train_multi": 0,
        "test_multi": 0,
        "test_multi_seen": 0,
        "test_multi_unseen1": 0,
        "test_multi_unseen2": 0,
        "exact_train_test_overlap": 0,
        "exact_train_multi_test_multi_overlap": 0,
    }
    datasets: dict[str, Any] = {}
    for ds, sp_any in sorted(split.items()):
        sp = dict(sp_any)
        train = [str(x) for x in sp.get("train", [])]
        test = [str(x) for x in sp.get("test", [])]
        train_multi = [x for x in train if is_multi(x)]
        test_multi = [str(x) for x in sp.get("test_multi", [])]
        overlap = sorted(set(train) & set(test))
        multi_overlap = sorted(set(train_multi) & set(test_multi))
        row = {
            "train": len(train),
            "test": len(test),
            "train_multi": len(train_multi),
            "test_multi": len(test_multi),
            "test_multi_seen": len(sp.get("test_multi_seen", [])),
            "test_multi_unseen1": len(sp.get("test_multi_unseen1", [])),
            "test_multi_unseen2": len(sp.get("test_multi_unseen2", [])),
            "exact_train_test_overlap": len(overlap),
            "exact_train_multi_test_multi_overlap": len(multi_overlap),
            "overlap_examples": overlap[:5],
            "multi_overlap_examples": multi_overlap[:5],
        }
        datasets[ds] = row
        totals["datasets"] += 1
        for key in (
            "train",
            "test",
            "train_multi",
            "test_multi",
            "test_multi_seen",
            "test_multi_unseen1",
            "test_multi_unseen2",
            "exact_train_test_overlap",
            "exact_train_multi_test_multi_overlap",
        ):
            totals[key] += int(row[key])
    return {"totals": totals, "datasets": datasets}


def make_arm(
    canonical: dict[str, Any],
    *,
    arm: str,
    recipe: dict[str, dict[str, int]],
    seed: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    split = copy.deepcopy(canonical)
    moved_by_dataset: dict[str, list[dict[str, str]]] = {}
    for ds, group_counts in recipe.items():
        if ds not in split:
            raise KeyError(f"{arm}: dataset not in canonical split: {ds}")
        sp = split[ds]
        moved: list[dict[str, str]] = []
        for group, n in group_counts.items():
            source = [str(x) for x in sp.get(group, [])]
            picked = stable_pick(source, n=n, seed=seed, arm=arm, dataset=ds, group=group)
            for cond in picked:
                moved.append({"condition": cond, "source_group": group})
        moved_conditions = [x["condition"] for x in moved]
        moved_set = set(moved_conditions)
        if moved_set & set(map(str, sp.get("train", []))):
            raise ValueError(f"{arm}/{ds}: picked condition already in train")
        sp["train"] = [str(x) for x in sp.get("train", [])] + moved_conditions
        for group in GROUPS_TO_REMOVE:
            sp[group] = [str(x) for x in sp.get(group, []) if str(x) not in moved_set]
        moved_by_dataset[ds] = moved

    audit = split_counts(split)
    if audit["totals"]["exact_train_test_overlap"] != 0:
        raise ValueError(f"{arm}: exact train/test overlap after split creation")
    if audit["totals"]["exact_train_multi_test_multi_overlap"] != 0:
        raise ValueError(f"{arm}: exact train multi/test multi overlap after split creation")
    metadata = {
        "arm": arm,
        "seed": seed,
        "recipe": recipe,
        "moved_by_dataset": moved_by_dataset,
        "audit": audit,
        "interpretation": (
            "few-shot diagnostic split; moved multi conditions are train supervision; "
            "remaining held-out multi conditions must be interpreted separately from zero-shot canonical evaluation"
        ),
    }
    return split, metadata


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Few-Shot Multi-Calibration Split Audit",
        "",
        f"Canonical split: `{payload['canonical_split']}`",
        f"Output directory: `{payload['out_dir']}`",
        f"Seed: `{payload['seed']}`",
        "",
        "These splits are diagnostic few-shot variants, not zero-shot promotion splits.",
        "All split JSONs contain only dataset keys so they are safe for `--split-file`.",
        "",
        "## Arms",
        "",
        "| Arm | moved multi | train multi | test multi | unseen2 held out | exact train/test overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in payload["arms"]:
        total_moved = sum(len(v) for v in arm["moved_by_dataset"].values())
        totals = arm["audit"]["totals"]
        lines.append(
            f"| `{arm['arm']}` | {total_moved} | {totals['train_multi']} | "
            f"{totals['test_multi']} | {totals['test_multi_unseen2']} | "
            f"{totals['exact_train_test_overlap']} |"
        )
    lines += ["", "## Moved Conditions", ""]
    for arm in payload["arms"]:
        lines.append(f"### `{arm['arm']}`")
        for ds, moved in sorted(arm["moved_by_dataset"].items()):
            lines.append(f"- `{ds}`: {len(moved)} moved")
            for row in moved:
                lines.append(f"  - `{row['condition']}` from `{row['source_group']}`")
        lines.append("")
    lines += [
        "## Gate",
        "",
        "- Held-out exact train/test overlap must remain `0`.",
        "- Wessels held-out unseen2 pp should improve by `+0.05` or turn positive.",
        "- Norman unseen2 should not regress by more than `0.03`.",
        "- Overall pp and family_gene pp should not clearly regress.",
        "- MMD clamped ratio should remain `<=1.15`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-split", type=Path, default=Path("/data/cyx/1030/dataset/biFlow_data/split_seed42.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data/cyx/1030/scLatent/runs/latentfm_fewshot_multi_calibration_20260621/splits"))
    parser.add_argument("--seed", default="20260621")
    parser.add_argument("--report-md", type=Path, default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_FEWSHOT_MULTI_CALIBRATION_SPLIT_AUDIT_20260621.md"))
    parser.add_argument("--report-json", type=Path, default=Path("/data/cyx/1030/scLatent/reports/latentfm_fewshot_multi_calibration_split_audit_20260621.json"))
    args = parser.parse_args()

    canonical = load_json(args.canonical_split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = []
    for arm, recipe in DEFAULT_ARMS.items():
        split, metadata = make_arm(canonical, arm=arm, recipe=recipe, seed=args.seed)
        split_path = args.out_dir / f"{arm}_split_seed42_fewshot20260621.json"
        meta_path = args.out_dir / f"{arm}_metadata.json"
        split_path.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        metadata["split_file"] = str(split_path)
        metadata["metadata_file"] = str(meta_path)
        arms.append(metadata)

    payload = {
        "canonical_split": str(args.canonical_split),
        "out_dir": str(args.out_dir),
        "seed": args.seed,
        "arms": arms,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"report_md": str(args.report_md), "report_json": str(args.report_json), "arms": list(DEFAULT_ARMS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
