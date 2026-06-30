#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORT = ROOT / "reports/LATENTFM_STRATEGY_ALL_DECISION_20260619.md"
CSV_OUT = ROOT / "reports/latentfm_strategy_all_decision_20260619.csv"
JSON_OUT = ROOT / "reports/latentfm_strategy_all_decision_20260619.json"

INPUTS = [
    ("four_run", ROOT / "reports/latentfm_strategy_probe_20260619.csv"),
    ("expanded", ROOT / "reports/latentfm_strategy_probe_expanded_20260619.csv"),
]

REFERENCE = {
    "scfoundation": {
        "name": "primary_scfoundation",
        "test_mmd": 0.027124,
        "test_pp": 0.0338,
        "family_gene_pp": 0.0437,
        "family_drug_pp": -0.0082,
        "multi_seen_pp": 0.2112,
        "multi_unseen1_pp": -0.0032,
        "multi_unseen2_pp": -0.1386,
        "resid_cosine": 0.0099,
        "resid_multi_seen_cosine": 0.0708,
        "resid_unseen1_cosine": 0.0006,
        "resid_unseen2_cosine": -0.2744,
    },
    "stack": {
        "name": "stack_comp006",
        "test_mmd": 0.039851,
        "test_pp": 0.0063,
        "family_gene_pp": 0.0133,
        "family_drug_pp": -0.0041,
        "multi_seen_pp": 0.1528,
        "multi_unseen1_pp": 0.0265,
        "multi_unseen2_pp": -0.0656,
        "resid_cosine": 0.0096,
        "resid_multi_seen_cosine": 0.1425,
        "resid_unseen1_cosine": 0.0258,
        "resid_unseen2_cosine": -0.1625,
    },
}


def input_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def fnum(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out = float(value)
        return out if math.isfinite(out) else None
    try:
        out = float(str(value))
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any, digits: int = 4) -> str:
    num = fnum(value)
    if num is None:
        return "NA"
    return f"{num:.{digits}f}"


def read_rows() -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for matrix, path in INPUTS:
        if not input_ready(path):
            missing.append(str(path))
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                row = dict(raw)
                row["matrix"] = matrix
                row["run"] = row.get("run") or row.get("short") or row.get("tag") or "NA"
                row["complete"] = str(row.get("complete", "")).lower() == "true"
                rows.append(row)
    return rows, missing


def composite(row: dict[str, Any]) -> float | None:
    existing = fnum(row.get("score"))
    if existing is not None:
        return existing
    test_pp = fnum(row.get("test_pp"))
    test_mmd = fnum(row.get("test_mmd"))
    unseen1 = fnum(row.get("multi_unseen1_pp"))
    unseen2 = fnum(row.get("multi_unseen2_pp"))
    gene = fnum(row.get("family_gene_pp"))
    resid_unseen2 = fnum(row.get("resid_unseen2_cosine"))
    if None in (test_pp, test_mmd, unseen1, unseen2, gene, resid_unseen2):
        return None
    return (
        float(test_pp)
        - 0.5 * float(test_mmd)
        + 0.4 * (float(unseen1) + float(unseen2))
        + 0.2 * float(gene)
        + 0.1 * float(resid_unseen2)
    )


def improvement(row: dict[str, Any], key: str) -> float | None:
    backbone = str(row.get("backbone") or "")
    ref = REFERENCE.get(backbone)
    value = fnum(row.get(key))
    if ref is None or value is None or key not in ref:
        return None
    return value - float(ref[key])


def mmd_ratio(row: dict[str, Any]) -> float | None:
    backbone = str(row.get("backbone") or "")
    ref = REFERENCE.get(backbone)
    value = fnum(row.get("test_mmd"))
    if ref is None or value is None:
        return None
    base = float(ref["test_mmd"])
    return value / base if base > 0 else None


def classify(row: dict[str, Any]) -> str:
    if not row.get("complete"):
        return "incomplete"
    score = composite(row)
    if score is None:
        return "incomplete"
    pp_gain = improvement(row, "test_pp")
    seen_gain = improvement(row, "multi_seen_pp")
    u1_gain = improvement(row, "multi_unseen1_pp")
    u2_gain = improvement(row, "multi_unseen2_pp")
    gene_gain = improvement(row, "family_gene_pp")
    ratio = mmd_ratio(row)
    required = (pp_gain, seen_gain, u1_gain, u2_gain, gene_gain, ratio)
    if any(v is None for v in required):
        return "needs_manual_review"
    if (
        pp_gain > 0
        and seen_gain > 0
        and u1_gain > 0
        and u2_gain > 0
        and gene_gain >= -0.002
        and ratio <= 1.10
    ):
        return "repeat_candidate"
    if (u1_gain > 0 or u2_gain > 0 or seen_gain > 0) and ratio <= 1.20:
        return "diagnostic_candidate"
    return "reject_as_is"


def normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    numeric_keys = [
        "iid_mmd",
        "iid_pc",
        "iid_pp",
        "test_mmd",
        "test_pc",
        "test_pp",
        "multi_seen_pp",
        "multi_unseen1_pp",
        "multi_unseen2_pp",
        "family_gene_pp",
        "family_drug_pp",
        "resid_cosine",
        "resid_multi_seen_cosine",
        "resid_unseen1_cosine",
        "resid_unseen2_cosine",
        "score",
    ]
    for row in rows:
        clean = {k: row.get(k) for k in row}
        for key in numeric_keys:
            clean[key] = fnum(row.get(key))
        clean["score"] = composite(clean)
        clean["mmd_ratio_to_ref"] = mmd_ratio(clean)
        for key in (
            "test_pp",
            "multi_seen_pp",
            "multi_unseen1_pp",
            "multi_unseen2_pp",
            "family_gene_pp",
            "family_drug_pp",
            "resid_unseen2_cosine",
        ):
            clean[f"delta_{key}"] = improvement(clean, key)
        clean["decision"] = classify(clean)
        out.append(clean)
    return out


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "matrix",
        "run",
        "backbone",
        "complete",
        "decision",
        "checkpoint_step",
        "test_mmd",
        "mmd_ratio_to_ref",
        "test_pc",
        "test_pp",
        "delta_test_pp",
        "multi_seen_pp",
        "delta_multi_seen_pp",
        "multi_unseen1_pp",
        "delta_multi_unseen1_pp",
        "multi_unseen2_pp",
        "delta_multi_unseen2_pp",
        "family_gene_pp",
        "delta_family_gene_pp",
        "family_drug_pp",
        "delta_family_drug_pp",
        "resid_cosine",
        "resid_unseen2_cosine",
        "delta_resid_unseen2_cosine",
        "score",
        "desc",
        "run_dir",
    ]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def write_report(rows: list[dict[str, Any]], missing: list[str]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    complete = [row for row in rows if row.get("complete") and row.get("score") is not None]
    ranked = sorted(complete, key=lambda r: float(r["score"]), reverse=True)
    repeat = [row for row in ranked if row.get("decision") == "repeat_candidate"]
    diagnostic = [row for row in ranked if row.get("decision") == "diagnostic_candidate"]
    status = "pending" if missing else "complete"
    present_inputs = [str(path) for _matrix, path in INPUTS if input_ready(path)]

    lines = [
        "# LatentFM Strategy All-Run Decision 2026-06-19",
        "",
        f"Generated: {datetime.now().strftime('%F %T')}",
        f"Status: `{status}`",
        "",
        "## Inputs",
        "",
    ]
    for matrix, path in INPUTS:
        lines.append(f"- `{matrix}`: `{path}` ({'present' if input_ready(path) else 'missing'})")
    lines += [
        "",
        "## Decision Rule",
        "",
        "A branch is a `repeat_candidate` only if it improves aggregate `test_pp`, `multi_seen_pp`, `multi_unseen1_pp`, and `multi_unseen2_pp` versus the matching backbone reference, keeps `family_gene_pp` approximately preserved, and keeps MMD within 10% of the reference. A branch with useful split-specific improvement but weaker preservation is a `diagnostic_candidate`, not a manuscript claim.",
        "",
        "## Reference Anchors",
        "",
        "| Backbone | Reference | MMD | pp | seen pp | unseen1 pp | unseen2 pp | gene pp | drug pp | resid unseen2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backbone, ref in REFERENCE.items():
        lines.append(
            f"| `{backbone}` | `{ref['name']}` | {fmt(ref['test_mmd'], 6)} | "
            f"{fmt(ref['test_pp'])} | {fmt(ref['multi_seen_pp'])} | "
            f"{fmt(ref['multi_unseen1_pp'])} | {fmt(ref['multi_unseen2_pp'])} | "
            f"{fmt(ref['family_gene_pp'])} | {fmt(ref['family_drug_pp'])} | "
            f"{fmt(ref['resid_unseen2_cosine'])} |"
        )
    lines += [
        "",
        "## Candidate Table",
        "",
        "| Run | Matrix | Backbone | Decision | MMD | MMD/ref | pp | d pp | seen d | unseen1 d | unseen2 d | gene d | drug d | resid unseen2 d | score |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| `{row.get('run')}` | `{row.get('matrix')}` | `{row.get('backbone')}` | "
            f"`{row.get('decision')}` | {fmt(row.get('test_mmd'), 6)} | "
            f"{fmt(row.get('mmd_ratio_to_ref'))} | {fmt(row.get('test_pp'))} | "
            f"{fmt(row.get('delta_test_pp'))} | {fmt(row.get('delta_multi_seen_pp'))} | "
            f"{fmt(row.get('delta_multi_unseen1_pp'))} | {fmt(row.get('delta_multi_unseen2_pp'))} | "
            f"{fmt(row.get('delta_family_gene_pp'))} | {fmt(row.get('delta_family_drug_pp'))} | "
            f"{fmt(row.get('delta_resid_unseen2_cosine'))} | {fmt(row.get('score'))} |"
        )

    lines += ["", "## Recommendation", ""]
    if missing:
        lines.append("Reports are still pending. Do not make a branch-selection decision yet.")
        if present_inputs and ranked:
            lines += [
                "",
                "Partial evidence:",
                "",
                f"- Present strategy CSV inputs: {len(present_inputs)} / {len(INPUTS)}.",
                f"- Complete candidate rows currently visible: {len(ranked)}.",
                "- These rows may be interpreted as diagnostics, but they are not the final 12-run gate.",
            ]
        lines += [
            "",
            "Next action:",
            "",
            (
                "Wait for the remaining upstream strategy CSVs and the watcher-generated full combined decision. "
                "Do not launch more LatentFM strategy jobs from a partial table."
                if present_inputs
                else "Wait for the posthoc watchers. Do not launch more LatentFM strategy jobs until at least one upstream strategy CSV exists."
            ),
        ]
    elif repeat:
        best = repeat[0]
        lines.append(
            f"Repeat/deepen `{best.get('run')}` first. It is the best strict repeat candidate under the current gate."
        )
        lines += [
            "",
            "Next action:",
            "",
            "1. Run at least one repeat seed with the same capped 4k-step budget to check that the gain is not seed noise.",
            "2. If the repeat preserves the gate, launch a longer 12k-20k run from the matching backbone checkpoint.",
            "3. Use `ops/select_available_gpus.py --samples 3 --interval-seconds 10 --need 1` before launching, and keep the three-jobs-per-physical-GPU LatentFM cap.",
            "4. Run full split/family/residual posthoc with ODE20 and condition-level top improved/failed tables before treating the branch as the formal LatentFM mainline.",
        ]
    elif diagnostic:
        best = diagnostic[0]
        lines.append(
            f"No strict repeat candidate yet. Best diagnostic branch is `{best.get('run')}`; use it to infer the next objective/architecture change before launching a long formal run."
        )
        lines += [
            "",
            "Next action:",
            "",
            "1. Do not scale this branch directly as a manuscript result.",
            "2. Inspect which split improved: multi-seen suggests memorized composition; unseen1/unseen2 suggests more useful combinatorial behavior; family loss suggests overfitting a perturbation type.",
            "3. Design one targeted follow-up that changes only the implicated mechanism, then run a short capped repeat before any long run.",
            "4. Keep the current reference anchors unchanged unless a new baseline is explicitly promoted and documented.",
        ]
    elif ranked:
        best = ranked[0]
        lines.append(
            f"All completed branches are `reject_as_is` under the current gate. Best score is `{best.get('run')}`, but it should not be promoted without a new objective or architecture change."
        )
        lines += [
            "",
            "Next action:",
            "",
            "1. Stop scalar loss-weight tuning in this objective family.",
            "2. Pivot to a qualitatively different condition-response alignment, architecture, or data-stratified training design.",
            "3. Treat the 12-run matrix as negative-control evidence in the audit trail, not as a failed endpoint.",
        ]
    else:
        lines.append("No complete rows are available yet.")
        lines += [
            "",
            "Next action:",
            "",
            "Wait for the posthoc watchers. Do not launch more LatentFM strategy jobs until at least one upstream strategy CSV exists.",
        ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    raw_rows, missing = read_rows()
    rows = normalize(raw_rows)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows)
    JSON_OUT.write_text(json.dumps({"missing_inputs": missing, "rows": rows}, indent=2), encoding="utf-8")
    write_report(rows, missing)
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
