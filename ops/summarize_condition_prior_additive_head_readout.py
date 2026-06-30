#!/usr/bin/env python3
"""Summarize condition-prior additive-head JSON into a Chinese decision note."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
ONE_SHOT = ROOT / "reports/LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_ONE_SHOT_STATUS_20260619.md"
ADD_JSON = ROOT / "reports/latentfm_condition_prior_additive_head_comparison_20260619.json"
ADD_REPORT = ROOT / "reports/LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_COMPARISON_20260619.md"
LAUNCH_REPORT = ROOT / "reports/目标推进阶段报告_20260619_1628.md"
OUT = ROOT / "reports/CONDITION_PRIOR_ADDITIVE_HEAD_READOUT_SUMMARY_20260619.md"


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


def main() -> int:
    payload = load_json(ADD_JSON)
    lines = [
        "# Condition-Prior Additive-Head Readout Summary 2026-06-19",
        "",
        f"Generated: {datetime.now().strftime('%F %T')}",
        "",
        "## Artifact State",
        "",
        f"- One-shot report: `{ONE_SHOT}` ({'present' if ONE_SHOT.is_file() else 'missing'})",
        f"- Additive-head report: `{ADD_REPORT}` ({'present' if ADD_REPORT.is_file() else 'missing'})",
        f"- Additive-head JSON: `{ADD_JSON}` ({'present' if ADD_JSON.is_file() else 'missing'})",
        f"- Launch report: `{LAUNCH_REPORT}` ({'present' if LAUNCH_REPORT.is_file() else 'missing'})",
        "",
    ]
    if payload is None:
        lines += [
            "## 结论",
            "",
            "Additive-head JSON 尚未生成。当前不能解释该分支，也不能启动后续 GPU 分支。",
            "",
            "下一步：等待训练/posthoc/summary watcher 或 30 分钟窗口 one-shot 生成 marker 和报告。",
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
    best = str(payload.get("best") or "NA")
    additive = next((row for row in rows if str(row.get("run")) == "scf_prioradd005_prior010_inject_e2_4k"), None)

    lines += [
        "## JSON Summary",
        "",
        f"- Status: `{status}`",
        f"- Complete rows: {len(complete)} / {len(rows)}",
        f"- Repeat candidates: {len(repeat)}",
        f"- Diagnostic candidates: {len(diagnostic)}",
        f"- Best: `{best}`",
        "",
        "## Rows",
        "",
        "| Run | Decision | Complete | MMD | pp | unseen1 pp | unseen2 pp | gene pp | score | add/combo Wessels unseen2 | add norm | int norm |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('run')}` | `{row.get('decision')}` | {row.get('complete')} | "
            f"{fmt(row.get('test_mmd'))} | {fmt(row.get('test_pp'))} | "
            f"{fmt(row.get('multi_unseen1_pp'))} | {fmt(row.get('multi_unseen2_pp'))} | "
            f"{fmt(row.get('family_gene_pp'))} | {fmt(row.get('score'))} | "
            f"{fmt(row.get('decomp_wessels_unseen2_combo_additive_cosine'))} | "
            f"{fmt(row.get('decomp_wessels_unseen2_additive_norm_ratio'))} | "
            f"{fmt(row.get('decomp_wessels_unseen2_interaction_norm_ratio'))} |"
        )
    lines.append("")

    lines += ["## 结论", ""]
    if pending:
        missing = "NA" if additive is None else str(additive.get("missing", "NA") or "NA")
        lines += [
            "Additive-head 分支仍未完成，当前不能解释效果，也不能把它计入 repeat/deepen 决策。",
            "",
            f"当前 additive 分支缺失产物：`{missing}`。",
            "",
            "下一步：等待训练/posthoc/summary watcher；如果训练或 posthoc EXIT_CODE 非 0，再按对应 RUN_STATUS 定位失败。",
        ]
    elif repeat:
        lines += [
            "至少一个分支达到 `repeat_candidate`。这仍不是最终成功；下一步只能对最佳 additive-head 配置做 repeat seed 或更深验证。",
            "",
            "需要同时检查 split/family 表和 decomposition 指标，确认 unseen2 改善不是以 MMD、seen、gene 或 additive/interaction 分解崩坏为代价。",
        ]
    elif additive and additive.get("decision") == "diagnostic_candidate":
        lines += [
            "Additive-head 分支只有 diagnostic 级别证据，没有达到 strict repeat-candidate gate。",
            "",
            "下一步不应继续调单个 additive loss 权重；应把结果用于决定是否实现更明确的 split-aware additive-plus-interaction 架构。",
        ]
    else:
        lines += [
            "Additive-head 分支未通过。应停止这一条单点机制，把它作为 negative/diagnostic evidence 记录。",
            "",
            "下一步优先整理失败模式，并转向更明确的 composition/intervention 表示设计。",
        ]
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)
    return 0 if not pending else 2


if __name__ == "__main__":
    raise SystemExit(main())
