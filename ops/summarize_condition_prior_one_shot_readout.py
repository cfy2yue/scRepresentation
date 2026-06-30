#!/usr/bin/env python3
"""Summarize condition-prior dose one-shot/dose JSON into a Chinese note."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
ONE_SHOT = ROOT / "reports/LATENTFM_CONDITION_PRIOR_DOSE_ONE_SHOT_STATUS_20260619.md"
DOSE_JSON = ROOT / "reports/latentfm_condition_prior_teacher_dose_20260619.json"
DOSE_REPORT = ROOT / "reports/LATENTFM_CONDITION_PRIOR_TEACHER_DOSE_20260619.md"
NEXT_ACTIONS = ROOT / "reports/LATENTFM_CONDITION_PRIOR_DOSE_NEXT_ACTIONS_20260619.md"
OUT = ROOT / "reports/CONDITION_PRIOR_DOSE_READOUT_SUMMARY_20260619.md"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def best_name(payload: dict[str, Any]) -> str:
    best = payload.get("best")
    if isinstance(best, dict):
        return str(best.get("run", "NA"))
    return str(best or "NA")


def main() -> int:
    payload = load_json(DOSE_JSON)
    lines = [
        "# Condition-Prior Dose Readout Summary 2026-06-19",
        "",
        f"Generated: {datetime.now().strftime('%F %T')}",
        "",
        "## Artifact State",
        "",
        f"- One-shot report: `{ONE_SHOT}` ({'present' if ONE_SHOT.is_file() else 'missing'})",
        f"- Dose report: `{DOSE_REPORT}` ({'present' if DOSE_REPORT.is_file() else 'missing'})",
        f"- Dose JSON: `{DOSE_JSON}` ({'present' if DOSE_JSON.is_file() else 'missing'})",
        f"- Decision playbook: `{NEXT_ACTIONS}`",
        "",
    ]
    if payload is None:
        lines += [
            "## 结论",
            "",
            "Dose JSON 尚未生成。当前不能推进 repeat/deepen，也不能判定 condition-prior teacher 是否有效。",
            "",
            "下一步：等待 watcher 或 one-shot 产物；不要手工重跑 posthoc，除非 RUN_STATUS/EXIT_CODE 明确显示失败。",
        ]
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(OUT)
        return 2

    rows = payload.get("rows") or []
    complete = [row for row in rows if row.get("complete")]
    repeat = [row for row in rows if row.get("decision") == "repeat_candidate"]
    diagnostic = [row for row in rows if row.get("decision") == "diagnostic_candidate"]
    pending = [row for row in rows if row.get("decision") == "pending" or not row.get("complete")]
    status = str(payload.get("status", "pending"))
    lines += [
        "## JSON Summary",
        "",
        f"- Status: `{status}`",
        f"- Complete rows: {len(complete)} / {len(rows)}",
        f"- Repeat candidates: {len(repeat)}",
        f"- Diagnostic candidates: {len(diagnostic)}",
        f"- Best: `{best_name(payload)}`",
        "",
        "## Dose Rows",
        "",
        "| Run | Decision | Complete | MMD | pp | unseen1 pp | unseen2 pp | gene pp | score |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('run')}` | `{row.get('decision')}` | {row.get('complete')} | "
            f"{fmt(row.get('test_mmd'))} | {fmt(row.get('test_pp'))} | "
            f"{fmt(row.get('multi_unseen1_pp'))} | {fmt(row.get('multi_unseen2_pp'))} | "
            f"{fmt(row.get('family_gene_pp'))} | {fmt(row.get('score'))} |"
        )
    lines.append("")

    lines += ["## 结论", ""]
    if repeat:
        lines += [
            "至少一个 dose 分支达到 `repeat_candidate`。这仍不是最终成功；下一步应只对最佳 dose 做 repeat/deepen，随后做更完整 posthoc 和 condition-level 生物解释表。",
            "",
            "建议先写：`reports/LATENTFM_CONDITION_PRIOR_REPEAT_PLAN_20260619.md`。",
        ]
    elif pending:
        lines += [
            "Dose 结果仍不完整。当前不能推进新 GPU 分支。",
            "",
            "下一步：等待 posthoc/dose-summary watcher；若 watcher EXIT_CODE 非 0，再按 RUN_STATUS 定位失败原因。",
        ]
    elif diagnostic:
        lines += [
            "没有 strict repeat candidate，但存在 diagnostic candidate。该结果只能作为机制证据，需要解释改善的是哪个指标轴。",
            "",
            "下一步优先写：`reports/LATENTFM_CONDITION_PRIOR_DIAGNOSTIC_INTERPRETATION_20260619.md`，再决定是否 pivot。",
        ]
    else:
        lines += [
            "所有 dose 分支均未通过。应停止 condition-prior scalar tuning，把结果作为 negative mechanistic evidence。",
            "",
            "下一步优先写：`reports/LATENTFM_CONDITION_PRIOR_NEGATIVE_RESULT_20260619.md`。",
        ]
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
