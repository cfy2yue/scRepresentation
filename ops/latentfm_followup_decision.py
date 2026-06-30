#!/usr/bin/env python3
"""Summarize LatentFM follow-up promotion status from existing JSON outputs.

This script is intentionally read-only. It does not inspect tmux, tail logs, or
launch jobs. Missing posthoc JSONs are reported as pending.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BASE = ROOT / "CoupledFM/output/latentfm_runs/full_scfoundation_alignment_smoke"
OUT = ROOT / "reports/LATENTFM_FOLLOWUP_DECISION_STATUS_20260617.md"
JSON_OUT = ROOT / "reports/latentfm_followup_decision_status_20260617.json"


@dataclass(frozen=True)
class RunSpec:
    label: str
    short: str
    tag: str
    require_complete: bool = True
    base: Path = BASE

    @property
    def run_dir(self) -> Path:
        return self.base / self.tag

    @property
    def iid(self) -> Path:
        return self.run_dir / "iid_eval_results.json"

    @property
    def split(self) -> Path:
        return self.run_dir / "posthoc_eval/split_group_eval_best_ode20_mse2048_mmd2048.json"

    @property
    def family(self) -> Path:
        return self.run_dir / "posthoc_eval/condition_family_eval_best_ode20_mse2048_mmd2048.json"

    @property
    def head(self) -> Path:
        return self.run_dir / "posthoc_eval/condition_delta_head_gene_test.json"


RUNS = [
    RunSpec(
        label="primary",
        short="primary",
        tag="20260617_scfoundation_comp006_delta_w5_12k",
        base=ROOT / "CoupledFM/output/latentfm_runs/full_scfoundation",
    ),
    RunSpec(
        label="head-only",
        short="head_only",
        tag="20260617_scfoundation_conddelta005_comp006_endpoint5_3k_smoke",
    ),
    RunSpec(
        label="head-injection",
        short="head_injection",
        tag="20260617_scfoundation_conddelta005_inject_comp006_endpoint5_3k_smoke",
    ),
    RunSpec(
        label="additive-atom",
        short="additive_atom",
        tag="20260617_scfoundation_conddelta005_addatom005_comp006_endpoint5_3k_smoke",
    ),
    RunSpec(
        label="pert-residual target",
        short="pert_residual_target",
        tag="20260617_scfoundation_conddelta005_pertresidtarget_comp006_endpoint5_3k_smoke",
        require_complete=False,
    ),
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def group_metric(doc: dict[str, Any] | None, group: str, key: str) -> float | None:
    if not doc:
        return None
    group_doc = doc.get("groups", {}).get(group, {})
    value = group_doc.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def metric(doc: dict[str, Any] | None, key: str) -> float | None:
    if not doc:
        return None
    value = doc.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def collect(spec: RunSpec) -> dict[str, Any]:
    iid = load_json(spec.iid)
    split = load_json(spec.split)
    family = load_json(spec.family)
    head = load_json(spec.head)
    missing = [
        str(path)
        for path in (spec.iid, spec.split, spec.family)
        if not path.is_file()
    ]
    return {
        "spec": spec,
        "iid": iid,
        "split": split,
        "family": family,
        "head": head,
        "missing": missing,
        "complete": not missing,
    }


def row_value(row: dict[str, Any], field: str) -> float | None:
    if field == "test_single":
        return group_metric(row["split"], "test_single", "pearson_pert")
    if field == "family_gene":
        return group_metric(row["family"], "family_gene", "pearson_pert")
    if field == "multi_unseen1":
        return group_metric(row["split"], "test_multi_unseen1", "pearson_pert")
    if field == "multi_unseen2":
        return group_metric(row["split"], "test_multi_unseen2", "pearson_pert")
    if field == "mmd":
        return metric(row["iid"], "test_mmd")
    if field == "head_spearman":
        return metric(row["head"], "pairwise_pred_target_cosine_spearman")
    raise KeyError(field)


def evaluate_candidate(primary: dict[str, Any], cand: dict[str, Any]) -> tuple[str, str]:
    if not cand["complete"]:
        return "pending", "missing split/family/IID JSONs"
    c = {k: row_value(cand, k) for k in ("test_single", "family_gene", "multi_unseen1", "multi_unseen2", "mmd", "head_spearman")}
    p = {k: row_value(primary, k) for k in ("test_single", "family_gene", "multi_unseen1", "multi_unseen2", "mmd")}
    if any(c[k] is None for k in ("test_single", "family_gene", "multi_unseen1", "multi_unseen2", "mmd")):
        return "incomplete", "required metrics are missing"
    improves_single_or_gene = (
        (p["test_single"] is not None and c["test_single"] >= p["test_single"])
        or (p["family_gene"] is not None and c["family_gene"] >= p["family_gene"])
    )
    preserves_multi = (
        p["multi_unseen1"] is not None
        and p["multi_unseen2"] is not None
        and c["multi_unseen1"] >= p["multi_unseen1"]
        and c["multi_unseen2"] >= p["multi_unseen2"]
    )
    mmd_ok = c["mmd"] <= 0.028
    strong = (
        c["multi_unseen1"] >= 0.0
        and c["multi_unseen2"] > -0.10
        and (c["family_gene"] >= 0.05 or c["test_single"] >= 0.06)
        and mmd_ok
    )
    if strong:
        return "strong_promote", "passes strong multi-unseen and single/gene gate"
    if improves_single_or_gene and preserves_multi and mmd_ok:
        return "promising", "passes practical promotion gate"
    if improves_single_or_gene and not preserves_multi:
        return "reject_as_is", "single/gene signal present but multi-unseen regresses versus primary"
    if not improves_single_or_gene:
        return "reject_as_is", "does not preserve primary single/gene PP"
    return "reject_as_is", "does not satisfy promotion gate"


def main() -> int:
    rows = [collect(spec) for spec in RUNS]
    primary = next(row for row in rows if row["spec"].short == "primary")
    lines = [
        "# LatentFM Follow-Up Decision Status",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Read-only decision summary from existing JSON outputs. This script does not inspect tmux or launch jobs.",
        "",
        "## Gate Table",
        "",
        "| Run | status | reason | test_single pp | family_gene pp | multi_unseen1 pp | multi_unseen2 pp | IID MMD | head Spearman |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    decisions: list[tuple[str, str, str]] = []
    json_rows: list[dict[str, Any]] = []
    for row in rows:
        spec = row["spec"]
        if not row["complete"] and not spec.require_complete and row["iid"] is None:
            continue
        if spec.short == "primary":
            status, reason = "reference", "primary scFoundation branch"
        else:
            status, reason = evaluate_candidate(primary, row)
        decisions.append((spec.short, status, reason))
        json_rows.append({
            "label": spec.label,
            "short": spec.short,
            "tag": spec.tag,
            "status": status,
            "reason": reason,
            "complete": bool(row["complete"]),
            "missing": list(row["missing"]),
            "test_single_pp": row_value(row, "test_single"),
            "family_gene_pp": row_value(row, "family_gene"),
            "multi_unseen1_pp": row_value(row, "multi_unseen1"),
            "multi_unseen2_pp": row_value(row, "multi_unseen2"),
            "iid_mmd": row_value(row, "mmd"),
            "head_spearman": row_value(row, "head_spearman"),
        })
        lines.append(
            f"| `{spec.label}` | {status} | {reason} | "
            f"{fmt(row_value(row, 'test_single'))} | "
            f"{fmt(row_value(row, 'family_gene'))} | "
            f"{fmt(row_value(row, 'multi_unseen1'))} | "
            f"{fmt(row_value(row, 'multi_unseen2'))} | "
            f"{fmt(row_value(row, 'mmd'), 6)} | "
            f"{fmt(row_value(row, 'head_spearman'))} |"
        )
    pending = [
        row
        for row in rows
        if row["missing"] and (row["spec"].require_complete or row["iid"] is not None)
    ]
    lines.extend(["", "## Pending Files", ""])
    if pending:
        lines.extend(["| Run | Missing files |", "|---|---|"])
        for row in pending:
            missing = "<br>".join(f"`{p}`" for p in row["missing"])
            lines.append(f"| `{row['spec'].label}` | {missing} |")
    else:
        lines.append("No required follow-up files are pending.")

    non_reference = [d for d in decisions if d[0] not in {"primary", "head_only"}]
    if any(status in {"strong_promote", "promising"} for _, status, _ in non_reference):
        recommendation = "A follow-up passed the gate; inspect the best row before launching a longer branch."
    elif any(status == "pending" for _, status, _ in non_reference):
        recommendation = "Wait for pending follow-up posthoc before launching the prepared pert-residual target smoke."
    else:
        recommendation = "Current follow-ups do not pass; prepared pert-residual target smoke is the next short branch."
    lines.extend(["", "## Recommendation", "", recommendation])

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_OUT.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "markdown_report": str(OUT),
                "recommendation": recommendation,
                "rows": json_rows,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(OUT)
    print(JSON_OUT)
    print(recommendation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
