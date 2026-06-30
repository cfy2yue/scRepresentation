#!/usr/bin/env python3
"""CPU gate for no-harm calibration readiness with positive controls.

This is a stricter follow-up to the global no-harm positive-class inventory.
It asks whether the project has enough frozen canonical evidence to train or
use a no-harm promotion surrogate. Canonical single/family metrics are used only
as historical calibration/veto evidence, not for selecting any active
checkpoint.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
INVENTORY = REPORTS / "latentfm_global_noharm_positive_class_inventory_20260624.json"
SURROGATE = REPORTS / "latentfm_scaling_noharm_surrogate_v2_gate_20260624.json"
OUT_JSON = REPORTS / "latentfm_noharm_calibration_positive_controls_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_NOHARM_CALIBRATION_POSITIVE_CONTROLS_GATE_20260624.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_family(run: str) -> str:
    name = run.lower()
    if "cap120" in name and "gene" in name:
        return "count_gene_background"
    if "cap120" in name:
        return "count_cap120"
    if "cap60_resp" in name:
        return "response_normalized"
    if "cap60_replay" in name or "cap60_6k" in name:
        return "cap60_step_replay"
    if "protocol_cap60" in name:
        return "protocol_cap60"
    if "softvisit" in name or "soft_exposure" in name:
        return "soft_exposure"
    if "risk_row" in name or "cvar" in name:
        return "risk_row_cvar"
    if "targetpop" in name:
        return "trackc_targetpop"
    if "support_context_v2" in name:
        return "trackc_support_context_v2_noop"
    return "other"


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> int:
    inventory = load(INVENTORY)
    surrogate = load(SURROGATE) if SURROGATE.exists() else {}
    rows = inventory.get("rows") or []
    enriched: list[dict[str, Any]] = []
    for row in rows:
        run = str(row.get("run") or "")
        test_pp = row.get("test_single_pp_delta")
        family_pp = row.get("family_gene_pp_delta")
        test_harm = row.get("test_single_pp_p_harm")
        family_harm = row.get("family_gene_pp_p_harm")
        material_gain = max(float(test_pp or 0.0), float(family_pp or 0.0))
        noharm = row.get("class") in {"nontrivial_positive", "trivial_or_noop_noharm"}
        harmful = (
            (test_harm is not None and float(test_harm) > 0.35)
            or (family_harm is not None and float(family_harm) > 0.35)
        )
        if row.get("class") == "nontrivial_positive":
            calib_class = "material_noharm_positive"
        elif row.get("class") == "trivial_or_noop_noharm":
            calib_class = "safe_zero_positive_control"
        elif harmful:
            calib_class = "harmful_negative_control"
        else:
            calib_class = "ambiguous_unusable"
        enriched.append(
            {
                "run": run,
                "family": infer_family(run),
                "calibration_class": calib_class,
                "inventory_class": row.get("class"),
                "test_single_pp_delta": test_pp,
                "test_single_pp_p_harm": test_harm,
                "family_gene_pp_delta": family_pp,
                "family_gene_pp_p_harm": family_harm,
                "material_gain_proxy": material_gain,
                "path": row.get("path"),
                "reasons": row.get("reasons") or [],
            }
        )

    class_counts = Counter(row["calibration_class"] for row in enriched)
    family_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for row in enriched:
        family_by_class[row["calibration_class"]][row["family"]] += 1
    material = [row for row in enriched if row["calibration_class"] == "material_noharm_positive"]
    safe_zero = [row for row in enriched if row["calibration_class"] == "safe_zero_positive_control"]
    harmful = [row for row in enriched if row["calibration_class"] == "harmful_negative_control"]
    positive_families = {row["family"] for row in material}
    negative_families = {row["family"] for row in harmful}

    reasons: list[str] = []
    if len(material) < 3:
        reasons.append("material_noharm_positive_count_lt_3")
    if len(positive_families) < 2:
        reasons.append("material_positive_family_count_lt_2")
    if len(safe_zero) < 2:
        reasons.append("safe_zero_positive_controls_lt_2")
    if len(harmful) < 6:
        reasons.append("harmful_negative_controls_lt_6")
    if len(negative_families) < 3:
        reasons.append("negative_family_count_lt_3")
    if len(material) == 0:
        reasons.append("promotion_surrogate_has_no_material_positive_class")
    if surrogate.get("status") and not str(surrogate.get("status")).endswith("pass"):
        reasons.append(f"prior_surrogate_status_{surrogate.get('status')}")

    status = (
        "noharm_calibration_positive_controls_pass_external_review_next"
        if not reasons
        else "noharm_calibration_positive_controls_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "historical_frozen_canonical_single_family_only": True,
            "uses_for_current_checkpoint_selection": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "inputs": {
            "inventory": str(INVENTORY),
            "surrogate_v2": str(SURROGATE),
        },
        "summary": {
            "n_rows": len(enriched),
            "class_counts": dict(class_counts),
            "family_counts_by_class": {k: dict(v) for k, v in family_by_class.items()},
            "n_material_noharm_positive": len(material),
            "n_safe_zero_positive_control": len(safe_zero),
            "n_harmful_negative_control": len(harmful),
            "n_material_positive_families": len(positive_families),
            "n_negative_families": len(negative_families),
            "prior_surrogate_status": surrogate.get("status"),
            "prior_surrogate_spearman_internal_score_vs_pp_harm": (surrogate.get("summary") or {}).get("spearman_internal_score_vs_pp_harm"),
        },
        "decision": {
            "can_train_promotion_surrogate": False if reasons else True,
            "can_authorize_gpu": False,
            "allowed_use": "veto_only" if reasons else "external_review_before_any_gpu",
        },
        "reasons": reasons,
        "rows": enriched,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    top = sorted(
        enriched,
        key=lambda row: (
            row["calibration_class"] == "material_noharm_positive",
            row["calibration_class"] == "safe_zero_positive_control",
            float(row.get("material_gain_proxy") or -999),
        ),
        reverse=True,
    )
    lines = [
        "# LatentFM No-Harm Calibration Positive Controls Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only readiness gate for a no-harm promotion surrogate.",
        "- Uses frozen canonical single/family no-harm decisions only as historical calibration/veto evidence.",
        "- Does not select/promote an active checkpoint, read canonical multi, read Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- rows: `{len(enriched)}`",
        f"- material no-harm positives: `{len(material)}`",
        f"- safe zero positive controls: `{len(safe_zero)}`",
        f"- harmful negative controls: `{len(harmful)}`",
        f"- material positive families: `{len(positive_families)}`",
        f"- negative families: `{len(negative_families)}`",
        f"- prior surrogate status: `{surrogate.get('status')}`",
        f"- prior internal-score vs pp-harm Spearman: `{fmt((surrogate.get('summary') or {}).get('spearman_internal_score_vs_pp_harm'))}`",
        "",
        "## Top Calibration Rows",
        "",
        "| class | family | run | test pp | test p_harm | family pp | family p_harm |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in top[:20]:
        lines.append(
            f"| `{row['calibration_class']}` | `{row['family']}` | `{row['run']}` | {fmt(row.get('test_single_pp_delta'))} | {fmt(row.get('test_single_pp_p_harm'))} | {fmt(row.get('family_gene_pp_delta'))} | {fmt(row.get('family_gene_pp_p_harm'))} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            "- GPU authorized: `False`",
            "- Current no-harm evidence can veto repeated unsafe scaling families, but cannot promote a new scaling GPU slate.",
            "- Reopen only after a genuinely new mechanism creates material no-harm positive examples under frozen canonical single/family evaluation.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "reasons": reasons}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
