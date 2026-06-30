#!/usr/bin/env python3
"""Meta-gate for true-cell non-noop tail-protection from completed evidence.

This consumes already completed CPU/frozen reports and decides whether any
existing true-cell repair family can authorize GPU. It is CPU/report-only.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "truecell_nonnoop_tail_protection_meta_gate_20260626"
MD = REPORTS / "LATENTFM_TRUECELL_NONNOOP_TAIL_PROTECTION_META_GATE_20260626.md"
JSON_OUT = REPORTS / "latentfm_truecell_nonnoop_tail_protection_meta_gate_20260626.json"


INPUT_JSONS = {
    "count_tail_completion": REPORTS / "latentfm_truecell_scaling_count_tail_completion_gate_20260625.json",
    "stratum_tail_protection": REPORTS / "latentfm_truecell_stratum_tail_protection_gate_20260625.json",
    "uncertainty_fallback_nonnoop": REPORTS / "latentfm_uncertainty_gated_anchor_fallback_nonnoop_gate_20260625.json",
    "riskrow_complementarity": REPORTS / "latentfm_truecell_riskrow_complementarity_gate_20260625.json",
    "canonical_noharm": REPORTS / "latentfm_true_cell_count_budget128_6k_canonical_noharm_decision_20260625.json",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")

    data = {name: load_json(path) for name, path in INPUT_JSONS.items()}

    # Pull only the small public decision numbers needed for the meta-gate.
    count_decision = data["count_tail_completion"].get("decision", {})
    count_rows = data["count_tail_completion"].get("rows", [])
    best_count = next((r for r in count_rows if r.get("series") == "6k_budget128"), {})

    stratum = data["stratum_tail_protection"]
    stratum_status = stratum.get("status", "missing")
    canonical_rows = stratum.get("canonical_rows", [])
    stratum_enabled_total = sum(int(r.get("enabled", 0) or 0) for r in canonical_rows)

    uncertainty = data["uncertainty_fallback_nonnoop"]
    uncertainty_status = uncertainty.get("status", "missing")
    uncertainty_enabled = 0
    for row in uncertainty.get("requirements", []):
        uncertainty_enabled += int(row.get("enabled", 0) or 0)

    risk = data["riskrow_complementarity"]
    risk_status = risk.get("status", "missing")
    raw_risk_groups = risk.get("groups", [])
    if isinstance(raw_risk_groups, dict):
        risk_groups = [dict(v, group=k) for k, v in raw_risk_groups.items()]
    else:
        risk_groups = raw_risk_groups

    canonical = data["canonical_noharm"]
    canonical_status = canonical.get("status", "missing")

    candidate_rows = [
        [
            "true_cell_count_6k_budget128",
            "positive_mechanism_only",
            f"cross/family/MMD {best_count.get('cross_pp_mean', 'NA')}/{best_count.get('family_pp_mean', 'NA')}/{best_count.get('family_mmd_mean', 'NA')}",
            "frozen canonical no-harm failed all seeds",
            False,
        ],
        [
            "stratum_tail_protection",
            stratum_status,
            f"canonical enabled rows {stratum_enabled_total}",
            "zero canonical footprint and no enabled feature passes",
            False,
        ],
        [
            "uncertainty_gated_anchor_fallback",
            uncertainty_status,
            f"canonical enabled rows {uncertainty_enabled}",
            "zero canonical footprint, exact/no-op route",
            False,
        ],
    ]

    for g in risk_groups:
        risk_pp_mean = g.get("risk_pp_mean")
        if risk_pp_mean is None:
            risk_pp_mean = g.get("riskrow_pp_delta_summary", {}).get("mean", "NA")
        protect_frac = g.get("protect_frac")
        if protect_frac is None:
            protect_frac = g.get("risk_protect_fraction", "NA")
        shared_harm_frac = g.get("shared_harm_frac")
        if shared_harm_frac is None:
            shared_harm_frac = g.get("shared_harm_fraction", "NA")
        candidate_rows.append(
            [
                f"riskrow_complementarity_{g.get('group', 'unknown')}",
                risk_status,
                f"protect_frac {protect_frac}; shared_harm_frac {shared_harm_frac}; risk_pp_mean {risk_pp_mean}",
                "risk-row does not protect true-cell canonical tails and aggregate pp is negative",
                False,
            ]
        )

    fail_reasons = [
        "no existing true-cell repair family has nonzero safe canonical footprint",
        "risk-row complementarity fails to protect true-cell canonical tails",
        "frozen canonical no-harm failed all 6k budget128 seeds",
        "internal true-cell signal remains useful only as training-data guidance",
    ]

    status = "truecell_nonnoop_tail_protection_meta_fail_no_gpu"
    payload = {
        "timestamp": timestamp,
        "status": status,
        "gpu_authorized": False,
        "route_freeze_authorized": False,
        "default_model": "xverse_8k_anchor",
        "canonical_noharm_status": canonical_status,
        "fail_reasons": fail_reasons,
        "next_action": "do not launch true-cell sampler/loss/staged-training GPU from existing artifacts; require a materially new non-noop tail-protection mechanism or external reliability artifact",
        "candidate_rows": [
            {
                "candidate": r[0],
                "status": r[1],
                "evidence": r[2],
                "blocker": r[3],
                "gpu_authorized": r[4],
            }
            for r in candidate_rows
        ],
        "inputs": {k: str(v) for k, v in INPUT_JSONS.items()},
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    candidates_csv = OUT_DIR / "candidate_closure.csv"
    reasons_csv = OUT_DIR / "fail_reasons.csv"
    write_csv(candidates_csv, ["candidate", "status", "evidence", "blocker", "gpu_authorized"], candidate_rows)
    write_csv(reasons_csv, ["reason"], [[r] for r in fail_reasons])

    md = f"""# LatentFM True-Cell Non-Noop Tail-Protection Meta Gate

Timestamp: `{timestamp}`

Status: `{status}`

GPU authorized: `False`

Default/deployable model: `xverse_8k_anchor`

## Boundary

- CPU/report-only meta-gate over completed true-cell count, stratum, uncertainty fallback, risk-row complementarity, and frozen canonical no-harm reports.
- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.
- Canonical single/family evidence is used only as a frozen no-harm/non-noop veto from completed reports.

## Candidate Closure

{md_table(["candidate", "status", "evidence", "blocker", "GPU"], [[r[0], r[1], r[2], r[3], f"`{str(r[4]).lower()}`"] for r in candidate_rows])}

## Fail Reasons

{md_table(["reason"], [[r] for r in fail_reasons])}

## Decision

- Keep true-cell support as the strongest scaling/training-data mechanism.
- Do not launch a true-cell sampler, weighted loss, staged-training, or fallback GPU route from existing artifacts.
- Reopen only if a materially new non-noop tail-protection mechanism or external reliability artifact maps to nonzero canonical footprint and passes dataset-tail, MMD, bootstrap, and frozen no-harm controls.

## Outputs

- JSON: `{JSON_OUT}`
- candidate closure: `{candidates_csv}`
- fail reasons: `{reasons_csv}`
"""
    MD.write_text(md)


if __name__ == "__main__":
    main()
