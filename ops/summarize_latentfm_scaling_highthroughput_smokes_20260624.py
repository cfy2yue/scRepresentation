#!/usr/bin/env python3
"""Summarize high-throughput LatentFM scaling exploratory smokes.

This decision layer is intentionally internal/train-only first. It does not
read canonical split metrics, canonical multi, or Track C query artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = Path(
    os.environ.get(
        "LATENTFM_SCALING_HT_RUN_ROOT",
        str(ROOT / "runs/latentfm_scaling_highthroughput_smokes_20260624"),
    )
)
OUT_JSON = Path(
    os.environ.get(
        "LATENTFM_SCALING_HT_DECISION_JSON",
        str(ROOT / "reports/latentfm_scaling_highthroughput_smokes_decision_20260624.json"),
    )
)
OUT_MD = Path(
    os.environ.get(
        "LATENTFM_SCALING_HT_DECISION_MD",
        str(ROOT / "reports/LATENTFM_SCALING_HIGH_THROUGHPUT_SMOKES_DECISION_20260624.md"),
    )
)

RUNS = [
    {
        "name": "xverse_scaling_cap60_6k_seed42",
        "role": "step_extension",
        "seed": 42,
        "steps": 6000,
        "anchor_replay_weight": 0.0,
    },
    {
        "name": "xverse_scaling_cap60_6k_seed43",
        "role": "seed_robustness",
        "seed": 43,
        "steps": 6000,
        "anchor_replay_weight": 0.0,
    },
    {
        "name": "xverse_scaling_cap60_replay05_4k_seed42",
        "role": "noharm_repair",
        "seed": 42,
        "steps": 4000,
        "anchor_replay_weight": 0.5,
    },
]

CUSTOM_RUNS = [item.strip() for item in os.environ.get("LATENTFM_SCALING_HT_RUNS", "").split(",") if item.strip()]
if CUSTOM_RUNS:
    meta = {
        "xverse_scaling_cap60_6k_seed42": {
            "role": "step_extension",
            "seed": 42,
            "steps": 6000,
            "anchor_replay_weight": 0.0,
        },
        "xverse_scaling_cap60_6k_seed43": {
            "role": "seed_robustness",
            "seed": 43,
            "steps": 6000,
            "anchor_replay_weight": 0.0,
        },
        "xverse_scaling_cap60_replay05_4k_seed42": {
            "role": "noharm_repair",
            "seed": 42,
            "steps": 4000,
            "anchor_replay_weight": 0.5,
        },
        "xverse_scaling_cap60_noot_3k_seed42": {
            "role": "noot_interaction",
            "seed": 42,
            "steps": 3000,
            "anchor_replay_weight": 0.0,
        },
        "xverse_scaling_cap60_noot_replay05_4k_seed42": {
            "role": "noot_replay_interaction",
            "seed": 42,
            "steps": 4000,
            "anchor_replay_weight": 0.5,
        },
        "xverse_scaling_cap60_6k_seed44": {
            "role": "seed_stability_confirmation",
            "seed": 44,
            "steps": 6000,
            "anchor_replay_weight": 0.0,
        },
        "xverse_scaling_pathway_quota12_3k_seed42": {
            "role": "pathway_quota_sampling",
            "seed": 42,
            "steps": 3000,
            "anchor_replay_weight": 0.0,
        },
        "xverse_scaling_pathway_randomcount_3k_seed42": {
            "role": "pathway_randomcount_control",
            "seed": 42,
            "steps": 3000,
            "anchor_replay_weight": 0.0,
        },
        "xverse_scaling_pathway_mmdpreserve_3k_seed42": {
            "role": "pathway_mmd_preservation",
            "seed": 42,
            "steps": 3000,
            "anchor_replay_weight": 0.0,
        },
    }
    RUNS = [
        {
            "name": name,
            **meta.get(
                name,
                {
                    "role": "custom",
                    "seed": None,
                    "steps": None,
                    "anchor_replay_weight": 0.0,
                },
            ),
        }
        for name in CUSTOM_RUNS
    ]

REFERENCE = {
    "name": "xverse_scaling_protocol_cap60_primary19_3k_seed42",
    "report": str(ROOT / "reports/LATENTFM_SCALING_PROTOCOL_MATRIX_DECISION_20260624.md"),
    "canonical_report": str(ROOT / "reports/LATENTFM_SCALING_PROTOCOL_CANONICAL_NOHARM_DECISION_20260624.md"),
    "internal_cross_pp_delta_vs_anchor": 0.010495,
    "internal_family_gene_pp_delta_vs_anchor": 0.012273,
    "internal_family_gene_mmd_delta_vs_anchor": -0.000857,
    "canonical_cross_bg_pp_delta": -0.006441,
    "canonical_all_single_p_harm": 0.605,
    "canonical_family_gene_p_harm": 0.847,
}

THRESHOLDS = {
    "cross_pp_delta_vs_anchor_min": 0.010,
    "internal_family_pp_delta_vs_anchor_min": 0.008,
    "family_gene_pp_delta_floor": -0.005,
    "family_gene_mmd_delta_ceiling": 0.001,
    "replay_family_mmd_delta_ceiling": 0.0005,
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_exit(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def group(payload: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if not payload:
        return {}
    return dict(((payload.get("groups") or {}).get(name) or {}))


def delta(cand: dict[str, Any], anchor: dict[str, Any], metric: str) -> float | None:
    if cand.get(metric) is None or anchor.get(metric) is None:
        return None
    return float(cand[metric]) - float(anchor[metric])


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:+.6f}"
    return str(x)


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in RUNS:
        run_dir = RUN_ROOT / spec["name"]
        eval_dir = run_dir / "posthoc_eval_internal"
        split_anchor = load_json(eval_dir / "split_group_eval_anchor_internal_ode20.json")
        split_cand = load_json(eval_dir / "split_group_eval_candidate_internal_ode20.json")
        fam_anchor = load_json(eval_dir / "condition_family_eval_anchor_internal_ode20.json")
        fam_cand = load_json(eval_dir / "condition_family_eval_candidate_internal_ode20.json")
        train_exit = read_exit(run_dir / f"{spec['name']}.EXIT_CODE")
        posthoc_exit = read_exit(run_dir / "POSTHOC_EXIT_CODE")

        cross_a = group(split_anchor, "internal_val_cross_background_seen_gene_proxy")
        cross_c = group(split_cand, "internal_val_cross_background_seen_gene_proxy")
        internal_family_a = group(split_anchor, "internal_val_family_gene_proxy")
        internal_family_c = group(split_cand, "internal_val_family_gene_proxy")
        family_gene_a = group(fam_anchor, "family_gene")
        family_gene_c = group(fam_cand, "family_gene")

        status = "done" if train_exit == 0 and posthoc_exit == 0 else "pending_or_failed"
        if train_exit not in (None, 0) or posthoc_exit not in (None, 0):
            status = "failed"

        rows.append(
            {
                **spec,
                "run_dir": str(run_dir),
                "status": status,
                "train_exit": train_exit,
                "posthoc_exit": posthoc_exit,
                "metrics": {
                    "cross_pp_delta_vs_anchor": delta(cross_c, cross_a, "pearson_pert"),
                    "cross_candidate_pp": cross_c.get("pearson_pert"),
                    "cross_anchor_pp": cross_a.get("pearson_pert"),
                    "internal_family_pp_delta_vs_anchor": delta(
                        internal_family_c, internal_family_a, "pearson_pert"
                    ),
                    "internal_family_mmd_delta_vs_anchor": delta(
                        internal_family_c, internal_family_a, "test_mmd"
                    ),
                    "family_gene_pp_delta_vs_anchor": delta(
                        family_gene_c, family_gene_a, "pearson_pert"
                    ),
                    "family_gene_mmd_delta_vs_anchor": delta(
                        family_gene_c, family_gene_a, "test_mmd"
                    ),
                },
            }
        )
    return rows


def gate_row(row: dict[str, Any]) -> tuple[bool, list[str]]:
    if row["status"] != "done":
        return False, [row["status"]]
    m = row["metrics"]
    reasons: list[str] = []
    if (m.get("cross_pp_delta_vs_anchor") or -999.0) < THRESHOLDS["cross_pp_delta_vs_anchor_min"]:
        reasons.append("cross_pp_delta_vs_anchor_lt_0p010")
    if (m.get("internal_family_pp_delta_vs_anchor") or -999.0) < THRESHOLDS["internal_family_pp_delta_vs_anchor_min"]:
        reasons.append("internal_family_pp_delta_vs_anchor_lt_0p008")
    if (m.get("family_gene_pp_delta_vs_anchor") or -999.0) < THRESHOLDS["family_gene_pp_delta_floor"]:
        reasons.append("family_gene_pp_hard_harm")
    mmd_ceiling = (
        THRESHOLDS["replay_family_mmd_delta_ceiling"]
        if row["anchor_replay_weight"] > 0
        else THRESHOLDS["family_gene_mmd_delta_ceiling"]
    )
    if (m.get("family_gene_mmd_delta_vs_anchor") or 999.0) > mmd_ceiling:
        reasons.append("family_gene_mmd_hard_harm")
    return not reasons, reasons


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(r["status"] == "failed" for r in rows):
        return {
            "status": "failed",
            "action": "inspect_failed_run_logs_once_then_fix_or_close",
            "passed": [],
            "failed": [r["name"] for r in rows if r["status"] == "failed"],
        }
    if any(r["status"] != "done" for r in rows):
        return {
            "status": "pending",
            "action": "wait_1800s_or_work_on_parallel_branches",
            "passed": [],
            "failed": [],
        }

    passed = []
    failed = []
    for row in rows:
        ok, reasons = gate_row(row)
        if ok:
            passed.append(row["name"])
        else:
            failed.append({"name": row["name"], "reasons": reasons})

    seed_rows = [r for r in rows if r["role"] in {"step_extension", "seed_robustness"}]
    seed_pass = [r["name"] for r in seed_rows if r["name"] in passed]
    replay_pass = [r["name"] for r in rows if r["role"] == "noharm_repair" and r["name"] in passed]
    if len(seed_pass) == 2 and replay_pass:
        status = "internal_pass_seed_and_replay"
        action = "freeze_passed_candidates_for_canonical_noharm_after_review"
    elif passed:
        status = "internal_partial_pass"
        action = "review_passed_candidate_then_choose_one_canonical_noharm_or_mutate_repair"
    else:
        status = "internal_fail"
        action = "close_failed_internal_smokes_and_prioritize_distinct_repair_branches"

    return {
        "status": status,
        "action": action,
        "passed": passed,
        "failed": failed,
        "thresholds": THRESHOLDS,
    }


def main() -> int:
    rows = collect_rows()
    decision = decide(rows)
    payload = {
        "status": decision["status"],
        "decision": decision,
        "reference": REFERENCE,
        "boundary": {
            "train_selection": "train_only_internal",
            "canonical_metrics_read": False,
            "canonical_multi_selection": False,
            "trackc_query_read": False,
            "gpu_launched_by_this_script": False,
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Scaling High-Throughput Smokes Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Boundary",
        "",
        "- Summarizes the new high-throughput scaling exploratory smokes on train-only internal validation.",
        "- Does not read canonical metrics, canonical multi for selection, or Track C query.",
        "- This report is not a promotion claim; it decides whether any arm deserves frozen canonical no-harm.",
        "",
        "## Reference",
        "",
        f"- Previous cap60 3k internal cross pp delta: `{REFERENCE['internal_cross_pp_delta_vs_anchor']:+.6f}`",
        f"- Previous cap60 3k internal family pp delta: `{REFERENCE['internal_family_gene_pp_delta_vs_anchor']:+.6f}`",
        f"- Previous cap60 3k canonical cross-bg pp delta: `{REFERENCE['canonical_cross_bg_pp_delta']:+.6f}`",
        f"- Previous cap60 3k canonical all/family p_harm: `{REFERENCE['canonical_all_single_p_harm']:.3f}/{REFERENCE['canonical_family_gene_p_harm']:.3f}`",
        "",
        "## Rows",
        "",
        "| run | role | status | cross pp delta | internal family pp delta | family pp delta | family MMD delta |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| `{row['name']}` | `{row['role']}` | `{row['status']}` | "
            f"{fmt(m.get('cross_pp_delta_vs_anchor'))} | "
            f"{fmt(m.get('internal_family_pp_delta_vs_anchor'))} | "
            f"{fmt(m.get('family_gene_pp_delta_vs_anchor'))} | "
            f"{fmt(m.get('family_gene_mmd_delta_vs_anchor'))} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- passed: `{decision.get('passed')}`",
            f"- failed: `{decision.get('failed')}`",
            "",
            "## Output",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
