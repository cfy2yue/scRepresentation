#!/usr/bin/env python3
"""Matched background/type confound simulation gate for scaling.

CPU-only gate. It asks whether local metadata can support a clean
source-verified background/type scaling experiment without mostly learning
dataset identity. It reads metadata inventory, split counts, and completed
source-strata/matched-breadth reports only.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_scaling_matched_background_type_confound_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_MATCHED_BACKGROUND_TYPE_CONFOUND_GATE_20260624.md"


def load_json(name: str) -> dict[str, Any]:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c:
            p = c / total
            h -= p * math.log2(p)
    return h


def normalized_mi(rows: list[dict[str, Any]], a: str, b: str) -> float:
    ca = Counter(str(r.get(a) or "unknown") for r in rows)
    cb = Counter(str(r.get(b) or "unknown") for r in rows)
    cab = Counter((str(r.get(a) or "unknown"), str(r.get(b) or "unknown")) for r in rows)
    total = len(rows)
    if total == 0:
        return 0.0
    mi = 0.0
    for (av, bv), c in cab.items():
        p = c / total
        pa = ca[av] / total
        pb = cb[bv] / total
        mi += p * math.log2(p / (pa * pb))
    denom = math.sqrt(max(entropy(ca), 1e-12) * max(entropy(cb), 1e-12))
    return mi / denom if denom > 0 else 0.0


def main() -> int:
    inventory = load_json("latentfm_scaling_metainfo_inventory_20260624.json")
    source_gate = load_json("latentfm_scaling_source_verified_background_type_strata_gate_20260624.json")
    matched = load_json("latentfm_matched_dataset_breadth_gate_20260624.json")
    matrix = load_json("latentfm_scaling_protocol_matrix_decision_20260624.json")

    rows = []
    for row in inventory.get("rows") or []:
        ds = str(row.get("dataset") or "")
        if not ds:
            continue
        rows.append(
            {
                "dataset": ds,
                "background": str(row.get("cell_line_meta") or "unknown"),
                "type": str(row.get("perturbation_type") or "unknown"),
                "train_count": int(row.get("trainonly_crossbg_v2_train") or 0),
                "cap120_count": int(row.get("cap120_all_v2_train") or 0),
                "obs_cell_type_n_unique": int(row.get("obs_cell_type_n_unique") or 0),
                "obs_pathway_n_unique": int(row.get("obs_pathway_n_unique") or 0),
            }
        )

    bg_to_types: dict[str, set[str]] = defaultdict(set)
    type_to_bgs: dict[str, set[str]] = defaultdict(set)
    bg_to_ds: dict[str, set[str]] = defaultdict(set)
    type_to_ds: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        bg_to_types[row["background"]].add(row["type"])
        type_to_bgs[row["type"]].add(row["background"])
        bg_to_ds[row["background"]].add(row["dataset"])
        type_to_ds[row["type"]].add(row["dataset"])

    crossing = {
        "n_datasets": len(rows),
        "n_backgrounds": len(bg_to_types),
        "n_types": len(type_to_bgs),
        "backgrounds_with_ge2_types": {k: sorted(v) for k, v in bg_to_types.items() if len(v) >= 2},
        "types_with_ge2_backgrounds": {k: sorted(v) for k, v in type_to_bgs.items() if len(v) >= 2},
        "background_dataset_counts": {k: len(v) for k, v in sorted(bg_to_ds.items())},
        "type_dataset_counts": {k: len(v) for k, v in sorted(type_to_ds.items())},
        "nmi_background_type": normalized_mi(rows, "background", "type"),
    }

    # Existing source gate already evaluates whether cap120-cap30 is broad
    # across strata. Reuse it as effect-side evidence.
    bg_summary = source_gate.get("background_summary") or {}
    type_summary = source_gate.get("perturbation_type_summary") or {}
    negative_backgrounds = [
        name for name, item in bg_summary.items() if float(item.get("pp_delta_mean") or 0.0) < -0.02
    ]
    negative_types = [
        name for name, item in type_summary.items() if float(item.get("pp_delta_mean") or 0.0) < -0.02
    ]
    strong_positive_strata = [
        ("background", name)
        for name, item in bg_summary.items()
        if float(item.get("pp_delta_mean") or 0.0) >= 0.02
    ] + [
        ("type", name)
        for name, item in type_summary.items()
        if float(item.get("pp_delta_mean") or 0.0) >= 0.02
    ]

    # Minimal feasibility for a clean matched experiment: several crossed
    # backgrounds/types, no major negative tails, and prior matched breadth not
    # already failing.
    reasons = []
    if len(crossing["backgrounds_with_ge2_types"]) < 3:
        reasons.append("too_few_backgrounds_with_multiple_types")
    if len(crossing["types_with_ge2_backgrounds"]) < 3:
        reasons.append("too_few_types_with_multiple_backgrounds")
    if crossing["nmi_background_type"] > 0.70:
        reasons.append("background_type_metadata_too_confounded")
    if negative_backgrounds:
        reasons.append("existing_background_tail_harm")
    if negative_types:
        reasons.append("existing_type_tail_harm")
    if "fail" in str(source_gate.get("status")):
        reasons.append("source_verified_strata_gate_already_failed")
    if "fail" in str(matched.get("status") or (matched.get("decision") or {}).get("status")):
        reasons.append("matched_dataset_breadth_gate_already_failed")
    if len(strong_positive_strata) < 3:
        reasons.append("too_few_broad_positive_source_strata")

    status = "matched_background_type_confound_gate_fail_no_gpu"
    if not reasons:
        status = "matched_background_type_confound_gate_pass_split_builder_next"

    payload = {
        "status": status,
        "gpu_authorized": status.endswith("_next"),
        "boundary": {
            "reads_local_metainfo_inventory": True,
            "reads_completed_reports_only": True,
            "canonical_multi_selection": False,
            "trackc_query_read": False,
            "training_or_inference": False,
            "active_logs_read": False,
            "gpu": False,
        },
        "crossing": crossing,
        "effect_side": {
            "source_gate_status": source_gate.get("status"),
            "matched_breadth_status": matched.get("status") or (matched.get("decision") or {}).get("status"),
            "protocol_matrix_status": matrix.get("status") or (matrix.get("decision") or {}).get("status"),
            "negative_backgrounds": negative_backgrounds,
            "negative_types": negative_types,
            "strong_positive_strata": strong_positive_strata,
        },
        "reasons": reasons,
        "next_action": (
            "build a source-matched split with shuffled dataset/type controls"
            if status.endswith("_next")
            else "do not launch background/type scaling GPU; metadata/effect side remains confounded"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Scaling Matched Background/Type Confound Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only metadata and completed-report synthesis.",
        "- Does not read canonical multi, Track C query, active logs, train, infer, or use GPU.",
        "",
        "## Metadata Crossing",
        "",
        f"- datasets: `{crossing['n_datasets']}`",
        f"- backgrounds: `{crossing['n_backgrounds']}`",
        f"- perturbation types: `{crossing['n_types']}`",
        f"- backgrounds with >=2 types: `{crossing['backgrounds_with_ge2_types']}`",
        f"- types with >=2 backgrounds: `{crossing['types_with_ge2_backgrounds']}`",
        f"- normalized MI(background,type): `{fmt(crossing['nmi_background_type'])}`",
        "",
        "## Existing Effect-Side Evidence",
        "",
        f"- source strata gate: `{source_gate.get('status')}`",
        f"- matched breadth gate: `{payload['effect_side']['matched_breadth_status']}`",
        f"- negative backgrounds: `{negative_backgrounds}`",
        f"- negative types: `{negative_types}`",
        f"- strong positive strata: `{strong_positive_strata}`",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        f"- GPU authorized: `{payload['gpu_authorized']}`",
        f"- next action: `{payload['next_action']}`",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": payload["gpu_authorized"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
