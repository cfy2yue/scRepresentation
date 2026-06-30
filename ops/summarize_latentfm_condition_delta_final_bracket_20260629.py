#!/usr/bin/env python3
"""Summarize the final condition-delta trust-region bracket."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BASE_RUN = ROOT / "runs/latentfm_lookahead_trust_region_adapter_smoke_20260627"
RUNS = {
    "seed42_40accepted": BASE_RUN / "xverse_lookahead_trust_region_adapter_seed42_40accepted_20260629_0040",
    "seed43_160accepted_thr5e6": BASE_RUN / "xverse_lookahead_trust_region_adapter_seed43_160accepted_thr5e6_20260629_1016",
    "seed44_160accepted_thr1e5_foot1e6": BASE_RUN / "xverse_lookahead_trust_region_adapter_seed44_160accepted_thr1e5_foot1e6_20260629_1016",
}
OUT_DIR = ROOT / "reports/latentfm_condition_delta_final_bracket_20260629"
OUT_MD = OUT_DIR / "LATENTFM_CONDITION_DELTA_FINAL_BRACKET_20260629.md"
OUT_JSON = OUT_DIR / "latentfm_condition_delta_final_bracket_20260629.json"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def fmt(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def read_train_stats(run_dir: Path) -> dict[str, Any]:
    rows = []
    path = run_dir / "train_metrics.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("accepted", "")).lower() == "true":
                rows.append(row)
    footprints = [float(r["proj_footprint_mean_l2"]) for r in rows if r.get("proj_footprint_mean_l2")]
    material = [float(r["proj_material_row_frac"]) for r in rows if r.get("proj_material_row_frac")]
    task_deltas = [float(r["proj_task_delta"]) for r in rows if r.get("proj_task_delta")]
    return {
        "accepted_metric_rows": len(rows),
        "footprint_mean_l2_max": max(footprints) if footprints else None,
        "footprint_mean_l2_last": footprints[-1] if footprints else None,
        "material_row_frac_max": max(material) if material else None,
        "material_row_frac_last": material[-1] if material else None,
        "task_delta_min": min(task_deltas) if task_deltas else None,
        "task_delta_last": task_deltas[-1] if task_deltas else None,
    }


def pick_delta(gate: dict[str, Any], stratum: str, metric: str) -> dict[str, Any]:
    for row in gate.get("paired_deltas", []):
        if row.get("stratum") == stratum and row.get("metric") == metric:
            return row
    return {}


def summarize_run(name: str, run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "summary.json")
    canonical_dir = run_dir / "posthoc_eval_canonical"
    gate = read_json(canonical_dir / "single_background_candidate_gate.json")
    internal_summary = read_json(run_dir / "posthoc/internal_eval_vs_anchor_summary.json")
    provenance = read_json(run_dir / "posthoc/lookahead_trust_region_smoke_provenance_audit.json")
    exit_code = (canonical_dir / "EXIT_CODE").read_text(encoding="utf-8").strip() if (canonical_dir / "EXIT_CODE").exists() else ""
    train_stats = read_train_stats(run_dir)
    return {
        "name": name,
        "run_dir": str(run_dir),
        "exit_code": exit_code,
        "accepted": summary.get("accepted"),
        "attempts": summary.get("attempts"),
        "anchor_threshold": summary.get("anchor_threshold"),
        "min_footprint": summary.get("min_footprint"),
        "step_grid": summary.get("step_grid"),
        "max_noop_drift": summary.get("max_noop_drift"),
        "internal_status": internal_summary.get("status"),
        "provenance_status": provenance.get("status"),
        "canonical_status": gate.get("gate", {}).get("status"),
        "canonical_reasons": gate.get("gate", {}).get("reasons", []),
        "train_stats": train_stats,
        "canonical_metrics": {
            "all_test_single_pp": pick_delta(gate, "all_test_single", "pearson_pert"),
            "cross_background_seen_gene_pp": pick_delta(gate, "cross_background_seen_gene", "pearson_pert"),
            "globally_unseen_gene_pp": pick_delta(gate, "globally_unseen_gene", "pearson_pert"),
            "family_gene_pp": pick_delta(gate, "family_gene", "pearson_pert"),
            "family_gene_mmd": pick_delta(gate, "family_gene", "test_mmd_clamped"),
        },
        "canonical_report": str(canonical_dir / "LATENTFM_LOOKAHEAD_TRUST_REGION_CANONICAL_NOHARM_DECISION.md"),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(name, path) for name, path in RUNS.items()]
    pass_rows = [row for row in rows if row["canonical_status"] != "candidate_gate_fail_close_or_nearmiss"]
    status = "condition_delta_final_bracket_close_family_no_gpu" if not pass_rows else "condition_delta_final_bracket_has_candidate"
    payload = {
        "created_at": now_cst(),
        "status": status,
        "decision": "close condition-delta family unless a new mechanism is proposed" if not pass_rows else "review passing candidates",
        "rows": rows,
        "boundary": "posthoc_summary_only_no_training_no_inference_no_gpu_no_canonical_multi_no_trackc_query",
    }
    write_json(OUT_JSON, payload)

    lines = [
        "# LatentFM Condition-Delta Final Bracket",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only synthesis over completed trust-region condition-delta smokes.",
        "* No training, inference, GPU launch, canonical multi selection, or Track C query.",
        "* Canonical gate reports are the authoritative decision artifacts.",
        "",
        "## Bracket Summary",
        "",
        "| run | accepted | attempts | anchor threshold | min footprint | max footprint | max material row frac | canonical status | key reasons | cross pp delta | all pp delta | family pp delta |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["canonical_metrics"]
        train = row["train_stats"]
        reasons = ",".join(row["canonical_reasons"])
        lines.append(
            f"| `{row['name']}` | `{row['accepted']}` | `{row['attempts']}` | `{row['anchor_threshold']}` | "
            f"`{row['min_footprint']}` | `{fmt(train.get('footprint_mean_l2_max'))}` | "
            f"`{fmt(train.get('material_row_frac_max'))}` | `{row['canonical_status']}` | `{reasons}` | "
            f"`{fmt(metrics['cross_background_seen_gene_pp'].get('delta_mean'))}` | "
            f"`{fmt(metrics['all_test_single_pp'].get('delta_mean'))}` | "
            f"`{fmt(metrics['family_gene_pp'].get('delta_mean'))}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "* All three condition-delta checkpoints passed the internal/provenance route but failed or nearmissed frozen canonical Track A promotion.",
            "* Increasing accepted updates from 40 to 160 and relaxing/materializing the trust-region footprint did not produce a material cross-background gain.",
            "* The best visible movement was seed43 with tiny positive pp deltas, but cross-background pp delta remained far below `+0.02` and p_improve below `0.90`; family pp harm risk also failed.",
            "* Close this exact condition-delta family. A future GPU route needs a new mechanism, not another seed/threshold/accepted-count tweak.",
            "",
            "## Reports",
            "",
        ]
    )
    for row in rows:
        lines.append(f"* `{row['name']}` canonical report: `{row['canonical_report']}`")
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(OUT_MD)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
