#!/usr/bin/env python3
"""Aggregate GSE92742 strict train/gene outcome materialization decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
INPUTS = [
    ROOT
    / "reports/lincs_gse92742_train_gene_outcome_eval_20260627/xverse_truecell_budget128_vs_anchor.json",
    *sorted((ROOT / "reports/lincs_gse92742_train_gene_candidate_panel_20260627").glob("*.json")),
]
OUT_JSON = ROOT / "reports/latentfm_lincs_gse92742_outcome_panel_decision_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_LINCS_GSE92742_OUTCOME_PANEL_DECISION_20260627.md"


def label_from_path(path: Path, obj: dict[str, Any]) -> str:
    boundary = obj.get("boundary")
    if isinstance(boundary, dict):
        label = boundary.get("candidate_label")
        if isinstance(label, str) and label:
            return label
    return path.name.removesuffix(".json").removesuffix("_vs_anchor")


def main() -> int:
    rows: list[dict[str, Any]] = []
    missing = [str(path) for path in INPUTS if not path.is_file()]
    for path in INPUTS:
        if not path.is_file():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        summary = obj.get("summary", {})
        best = summary.get("best_pp_signal") if isinstance(summary, dict) else None
        rows.append(
            {
                "label": label_from_path(path, obj),
                "path": str(path),
                "status": obj.get("status"),
                "gpu_authorized": bool(obj.get("gpu_authorized")),
                "conditions": summary.get("conditions"),
                "datasets": summary.get("datasets"),
                "exact_background_conditions": summary.get("exact_background_conditions"),
                "mean_pp_delta": summary.get("mean_pp_delta"),
                "mean_mmd_delta": summary.get("mean_mmd_delta"),
                "best_pp_feature": best.get("feature") if isinstance(best, dict) else None,
                "best_pp_rho": best.get("rho") if isinstance(best, dict) else None,
                "best_pp_shuffle_p_abs": best.get("shuffle_p_abs") if isinstance(best, dict) else None,
                "max_abs_mmd_signal_rho": summary.get("max_abs_mmd_signal_rho"),
                "reasons": obj.get("reasons", []),
            }
        )

    passes = [
        row
        for row in rows
        if row["status"] == "lincs_gse92742_train_gene_outcome_eval_pass_review_only_no_gpu"
    ]
    positive_mean = [
        row
        for row in rows
        if isinstance(row.get("mean_pp_delta"), (int, float)) and float(row["mean_pp_delta"]) > 0
    ]
    best_mean = max(
        rows,
        key=lambda row: float(row["mean_pp_delta"])
        if isinstance(row.get("mean_pp_delta"), (int, float))
        else float("-inf"),
        default=None,
    )
    best_signal = max(
        rows,
        key=lambda row: abs(float(row["best_pp_rho"]))
        if isinstance(row.get("best_pp_rho"), (int, float))
        else float("-inf"),
        default=None,
    )

    status = (
        "lincs_gse92742_outcome_panel_has_review_only_pass_no_gpu"
        if passes
        else "lincs_gse92742_outcome_panel_all_fail_close_no_gpu"
    )
    decision = (
        "Send passing candidates to external/protocol review only; do not train "
        "or promote from this report."
        if passes
        else "Close the current GSE92742 strict train/gene outcome-panel route "
        "as a GPU launcher. Preserve cap60 mean-gain and negative signal-control "
        "evidence as mechanism context."
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "missing_inputs": missing,
        "n_candidates": len(rows),
        "n_review_only_pass": len(passes),
        "n_positive_mean_pp_delta": len(positive_mean),
        "best_mean_pp_delta": best_mean,
        "best_abs_lincs_signal": best_signal,
        "rows": rows,
        "decision": decision,
        "boundary": {
            "split": str(
                ROOT
                / "dataset/biFlow_data/split_seed42_lincs_gse92742_train_gene_eval_20260627.json"
            ),
            "training_or_checkpoint_selection_used": False,
            "canonical_multi_selection_used": False,
            "trackc_heldout_query_used": False,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fmt(value: Any, digits: int = 6) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):+.{digits}f}"
        return "`None`" if value is None else f"`{value}`"

    lines = [
        "# LatentFM LINCS GSE92742 Outcome Panel Decision",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Eval-only frozen checkpoint panel on the strict GSE92742 S0 train/gene split.",
        "- No training, checkpoint selection, canonical multi selection, or Track C held-out query.",
        "- Closed candidates remain closed for promotion; this panel is mechanism/source diagnostic only.",
        "",
        "## Summary",
        "",
        f"- candidates audited: `{len(rows)}`",
        f"- review-only passes: `{len(passes)}`",
        f"- candidates with positive mean pp delta: `{len(positive_mean)}`",
        f"- best mean pp delta: `{best_mean['label'] if best_mean else None}` {fmt(best_mean.get('mean_pp_delta') if best_mean else None)}",
        f"- best absolute LINCS pp signal: `{best_signal['label'] if best_signal else None}` "
        f"{fmt(best_signal.get('best_pp_rho') if best_signal else None)} "
        f"(shuffle p {best_signal.get('best_pp_shuffle_p_abs') if best_signal else None})",
        "",
        "## Candidate Rows",
        "",
        "| candidate | status | n | datasets | exact-bg | mean pp delta | mean MMD delta | best pp signal | shuffle p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['label']}`",
                    f"`{row['status']}`",
                    str(row.get("conditions")),
                    str(row.get("datasets")),
                    str(row.get("exact_background_conditions")),
                    fmt(row.get("mean_pp_delta")),
                    fmt(row.get("mean_mmd_delta")),
                    fmt(row.get("best_pp_rho")),
                    str(row.get("best_pp_shuffle_p_abs")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            decision,
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
