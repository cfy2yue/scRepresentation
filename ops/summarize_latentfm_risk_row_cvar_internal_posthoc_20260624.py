#!/usr/bin/env python3
"""Internal train-only no-harm summary for the risk-row CVaR smoke.

This reads only posthoc outputs generated on the train-only xverse split. It
does not read canonical split metrics, canonical multi, Track C query, or
held-out query artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_NAME = "xverse_risk_row_cvar_allrisk_w020_2k_seed42"
RUN_DIR = ROOT / "runs/latentfm_risk_row_cvar_trainonly_20260624" / RUN_NAME
EVAL_DIR = RUN_DIR / "posthoc_eval_internal"
MECH_JSON = ROOT / "reports/latentfm_risk_row_cvar_trainonly_smoke_decision_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_risk_row_cvar_internal_posthoc_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_RISK_ROW_CVAR_INTERNAL_POSTHOC_DECISION_20260624.md"
RISK_DATASETS = [
    "Nadig_hepg2",
    "Nadig_jurket",
    "NormanWeissman2019_filtered",
    "ReplogleWeissman2022_K562_gwps",
    "Replogle_RPE1essential",
    "TianActivation",
]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def metric(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None:
        return None
    return float(value)


def group_delta(anchor: dict[str, Any], cand: dict[str, Any], group: str) -> dict[str, Any]:
    a = (anchor.get("groups") or {}).get(group) or {}
    c = (cand.get("groups") or {}).get(group) or {}
    out: dict[str, Any] = {"group": group, "n_conds": c.get("n_conds")}
    for name in ("pearson_pert", "test_mmd", "direct_pearson", "pearson_ctrl", "test_mse"):
        av = metric(a, name)
        cv = metric(c, name)
        out[f"anchor_{name}"] = av
        out[f"candidate_{name}"] = cv
        out[f"delta_{name}"] = None if av is None or cv is None else cv - av
    return out


def risk_dataset_mmd(anchor: dict[str, Any], cand: dict[str, Any], group: str) -> list[dict[str, Any]]:
    a_group = (anchor.get("groups") or {}).get(group) or {}
    c_group = (cand.get("groups") or {}).get(group) or {}
    a_ds = a_group.get("per_ds_mmd") or {}
    c_ds = c_group.get("per_ds_mmd") or {}
    rows = []
    for ds in RISK_DATASETS:
        av = a_ds.get(ds)
        cv = c_ds.get(ds)
        rows.append(
            {
                "dataset": ds,
                "anchor_mmd": None if av is None else float(av),
                "candidate_mmd": None if cv is None else float(cv),
                "delta_mmd": None if av is None or cv is None else float(cv) - float(av),
            }
        )
    return rows


def main() -> int:
    mech = load(MECH_JSON)
    split_anchor = load(EVAL_DIR / "split_group_eval_anchor_internal_ode20.json")
    split_cand = load(EVAL_DIR / "split_group_eval_candidate_internal_ode20.json")
    fam_anchor = load(EVAL_DIR / "condition_family_eval_anchor_internal_ode20.json")
    fam_cand = load(EVAL_DIR / "condition_family_eval_candidate_internal_ode20.json")

    exit_code = (RUN_DIR / "POSTHOC_EXIT_CODE").read_text(encoding="utf-8").strip() if (RUN_DIR / "POSTHOC_EXIT_CODE").exists() else None

    split_rows = [
        group_delta(split_anchor, split_cand, g)
        for g in ("test", "test_single", "internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy")
    ]
    family_rows = [
        group_delta(fam_anchor, fam_cand, g)
        for g in ("test_all", "family_gene", "test_single")
    ]
    risk_rows = risk_dataset_mmd(split_anchor, split_cand, "test")

    by_group = {row["group"]: row for row in split_rows + family_rows}
    risk_harm_rows = [
        row for row in risk_rows
        if row["delta_mmd"] is not None and row["delta_mmd"] > 0.002
    ]

    checks = {
        "mechanism_activated": mech.get("status") == "risk_row_cvar_trainonly_smoke_mechanism_activated_no_promotion",
        "posthoc_exit_code_zero": exit_code == "0",
        "cross_pp_no_large_harm": (by_group["internal_val_cross_background_seen_gene_proxy"]["delta_pearson_pert"] or -999.0) >= -0.005,
        "family_proxy_pp_no_large_harm": (by_group["internal_val_family_gene_proxy"]["delta_pearson_pert"] or -999.0) >= -0.005,
        "test_single_pp_no_large_harm": (by_group["test_single"]["delta_pearson_pert"] or -999.0) >= -0.005,
        "family_gene_pp_no_large_harm": (by_group["family_gene"]["delta_pearson_pert"] or -999.0) >= -0.005,
        "family_gene_mmd_no_large_harm": (by_group["family_gene"]["delta_test_mmd"] or 999.0) <= 0.002,
        "risk_dataset_mmd_harm_count_ok": len(risk_harm_rows) <= 2,
    }

    if exit_code is None:
        status = "risk_row_cvar_internal_posthoc_running_no_decision"
    elif all(checks.values()):
        status = "risk_row_cvar_internal_posthoc_pass_no_promotion"
    else:
        status = "risk_row_cvar_internal_posthoc_fail_close_or_mutate"

    payload = {
        "status": status,
        "run_name": RUN_NAME,
        "exit_code": exit_code,
        "boundary": {
            "train_only_internal_split": True,
            "canonical_metrics_read": False,
            "canonical_multi_read": False,
            "trackc_query_read": False,
        },
        "checks": checks,
        "split_group_deltas": split_rows,
        "family_group_deltas": family_rows,
        "risk_dataset_mmd_rows": risk_rows,
        "risk_dataset_mmd_harm_rows": risk_harm_rows,
        "decision": {
            "promotion_authorized": False,
            "canonical_noharm_authorized": False,
            "next_if_pass": "External audit before any frozen canonical no-harm consideration.",
            "next_if_fail": "Close or mutate risk-row CVaR; record internal negative evidence.",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Risk-Row CVaR Internal Posthoc Decision",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- Train-only/internal split posthoc only.",
        "- No canonical split metrics, canonical multi, Track C query, or held-out query artifacts.",
        "",
        "## Checks",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for name, value in checks.items():
        lines.append(f"| `{name}` | `{bool(value)}` |")
    lines.extend(["", "## Key Deltas", "", "| group | delta pp | delta MMD | n |", "|---|---:|---:|---:|"])
    for row in split_rows + family_rows:
        lines.append(
            f"| `{row['group']}` | `{row['delta_pearson_pert']}` | `{row['delta_test_mmd']}` | `{row['n_conds']}` |"
        )
    lines.extend(["", "## Risk Dataset MMD", "", "| dataset | delta MMD |", "|---|---:|"])
    for row in risk_rows:
        lines.append(f"| `{row['dataset']}` | `{row['delta_mmd']}` |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- No promotion or canonical no-harm is authorized by this summary alone.",
            "- A pass only permits external review of whether a frozen canonical no-harm gate is worth considering.",
            "",
            "## Output",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
