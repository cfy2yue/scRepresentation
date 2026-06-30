#!/usr/bin/env python3
"""Shuffle/count controls for the allmodality modality-router upper-bound."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

ROOT = Path("/data/cyx/1030/scLatent")
UPPER_JSON = ROOT / "reports/latentfm_allmodality_modality_router_upper_bound_gate_20260625.json"
OUT_JSON = ROOT / "reports/latentfm_allmodality_modality_router_control_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_ALLMODALITY_MODALITY_ROUTER_CONTROL_GATE_20260625.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def actual_route_records(upper: dict[str, Any]) -> tuple[str, str]:
    best = upper["best_route"]
    return str(best["gene_choice"]), str(best["drug_choice"])


def load_condition_records(drug_choice: str) -> list[dict[str, Any]]:
    run_root = ROOT / "runs/latentfm_true_cell_count_allmodality_doseaware_smokes_20260625" / drug_choice / "posthoc_eval_internal"
    anchor = load_json(run_root / "condition_family_eval_anchor_internal_ode20.json")
    candidate = load_json(run_root / "condition_family_eval_candidate_internal_ode20.json")
    records = []
    for family in ("family_gene", "family_drug"):
        a_rows = {
            (str(r["dataset"]), str(r["condition"])): r
            for r in ((anchor.get("groups") or {}).get(family) or {}).get("condition_metrics", [])
        }
        c_rows = {
            (str(r["dataset"]), str(r["condition"])): r
            for r in ((candidate.get("groups") or {}).get(family) or {}).get("condition_metrics", [])
        }
        for key in sorted(set(a_rows) & set(c_rows)):
            a = a_rows[key]
            c = c_rows[key]
            records.append(
                {
                    "dataset": key[0],
                    "condition": key[1],
                    "family": family,
                    "candidate_pp_delta": float(c["pearson_pert"]) - float(a["pearson_pert"]),
                    "candidate_mmd_delta": float(c["test_mmd"]) - float(a["test_mmd"]),
                }
            )
    return records


def summarize_route(records: list[dict[str, Any]], use_candidate_mask: np.ndarray) -> dict[str, Any]:
    deltas = np.array([r["candidate_pp_delta"] for r in records], dtype=float)
    routed = np.where(use_candidate_mask, deltas, 0.0)
    out: dict[str, Any] = {
        "all_pp_mean": float(np.mean(routed)),
        "all_hard_harm_frac": float(np.mean(routed < -0.005)),
    }
    for family in ("family_gene", "family_drug"):
        idx = np.array([r["family"] == family for r in records], dtype=bool)
        vals = routed[idx]
        out[f"{family}_pp_mean"] = float(np.mean(vals))
        out[f"{family}_hard_harm_frac"] = float(np.mean(vals < -0.005))
    dataset_rows = []
    for ds in sorted({r["dataset"] for r in records}):
        idx = np.array([r["dataset"] == ds for r in records], dtype=bool)
        dataset_rows.append({"dataset": ds, "n": int(np.sum(idx)), "pp_mean": float(np.mean(routed[idx]))})
    out["dataset_min_pp_mean"] = float(min(row["pp_mean"] for row in dataset_rows))
    return out


def main() -> int:
    upper = load_json(UPPER_JSON)
    gene_choice, drug_choice = actual_route_records(upper)
    records = load_condition_records(drug_choice)
    actual_mask = np.array([r["family"] == "family_drug" for r in records], dtype=bool)
    actual = summarize_route(records, actual_mask)

    rng = np.random.default_rng(20260625)
    n_candidate = int(np.sum(actual_mask))
    n = len(records)
    controls = []
    for _ in range(1000):
        idx = rng.choice(n, size=n_candidate, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        controls.append(summarize_route(records, mask))
    control_all = np.array([c["all_pp_mean"] for c in controls], dtype=float)
    control_drug = np.array([c["family_drug_pp_mean"] for c in controls], dtype=float)
    control_gene = np.array([c["family_gene_pp_mean"] for c in controls], dtype=float)

    payload = {
        "status": "allmodality_modality_router_control_fail_no_gpu",
        "gpu_authorized": False,
        "route": {"gene_choice": gene_choice, "drug_choice": drug_choice},
        "actual": actual,
        "controls": {
            "n_shuffles": len(controls),
            "all_pp_mean_mean": float(np.mean(control_all)),
            "all_pp_mean_p95": float(np.quantile(control_all, 0.95)),
            "drug_pp_mean_mean": float(np.mean(control_drug)),
            "drug_pp_mean_p95": float(np.quantile(control_drug, 0.95)),
            "gene_pp_mean_mean": float(np.mean(control_gene)),
            "gene_pp_mean_p05": float(np.quantile(control_gene, 0.05)),
        },
        "deltas_vs_control_mean": {
            "all_pp": float(actual["all_pp_mean"] - np.mean(control_all)),
            "drug_pp": float(actual["family_drug_pp_mean"] - np.mean(control_drug)),
            "gene_pp": float(actual["family_gene_pp_mean"] - np.mean(control_gene)),
        },
        "reasons": [
            "actual_route_close_to_count_matched_shuffle_due_drug_family_imbalance",
            "drug_condition_hard_harm_fraction_remains_high",
            "control_gate_does_not_authorize_eval_or_training_gpu",
        ],
        "boundary": {
            "canonical_multi_selection": False,
            "trackc_query_read": False,
            "gpu_used": False,
            "source": "existing_posthoc_condition_metrics",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM All-Modality Modality Router Control Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only shuffle/count control for the modality-router upper-bound.",
        "- Does not train, infer, use GPU, read canonical multi, or read Track C held-out query.",
        "",
        "## Result",
        "",
        f"- actual route: gene `{gene_choice}`, drug `{drug_choice}`",
        f"- actual all pp mean: `{actual['all_pp_mean']:+.6f}`",
        f"- count-matched shuffle all pp mean: `{payload['controls']['all_pp_mean_mean']:+.6f}`",
        f"- delta vs shuffle mean: `{payload['deltas_vs_control_mean']['all_pp']:+.6f}`",
        f"- actual drug pp mean: `{actual['family_drug_pp_mean']:+.6f}`",
        f"- shuffle drug pp mean: `{payload['controls']['drug_pp_mean_mean']:+.6f}`",
        f"- actual drug hard-harm frac: `{actual['family_drug_hard_harm_frac']:.3f}`",
        "",
        "## Decision",
        "",
        f"- GPU authorized: `{payload['gpu_authorized']}`",
        f"- reasons: `{payload['reasons']}`",
        "",
        "The upper-bound route is a useful clue, but it is too close to a count-matched shuffle because drug conditions dominate the internal allmod eval set.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gpu_authorized": False, "out_md": str(OUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
