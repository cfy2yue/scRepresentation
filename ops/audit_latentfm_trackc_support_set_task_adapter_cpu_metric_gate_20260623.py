#!/usr/bin/env python3
"""Fail-closed CPU metric gate for the Track C support-set task adapter.

The protocol requires a distinct support-set/task input, train_multi fitting or
predeclared small-grid evidence, support_val_multi scoring, zero/shuffled
support controls, and no held-out query/canonical multi selection. This audit
checks whether those inputs exist now. It must not reuse the consumed
support-context/residual/FiLM artifacts as if they were support-set task
adapter metric evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
PROTOCOL_JSON = REPORTS / "latentfm_trackc_support_set_task_adapter_protocol_20260623.json"
CODE_JSON = REPORTS / "latentfm_trackc_support_set_task_adapter_code_boundary_20260623.json"
OLD_SUPPORT_ROOT = (
    ROOT
    / "runs/latentfm_trackc_anchor_gated_support_teacher_artifacts_20260623/"
    "xverse_support_film_retry1_condition_means_artifacts/condition_means"
)
OUT_JSON = REPORTS / "latentfm_trackc_support_set_task_adapter_cpu_metric_gate_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_SET_TASK_ADAPTER_CPU_METRIC_GATE_20260623.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_counts(split: dict[str, Any]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    by_dataset: dict[str, dict[str, int]] = {}
    for dataset, groups in split.items():
        if not isinstance(groups, dict):
            continue
        row = {
            key: len(groups.get(key) or [])
            for key in ("train_multi", "support_val_multi", "test_multi", "train_single")
        }
        by_dataset[str(dataset)] = row
        for key, value in row.items():
            totals[key] += int(value)
    focus = {
        ds: by_dataset.get(ds, {})
        for ds in ("NormanWeissman2019_filtered", "Wessels")
    }
    return {"totals": dict(totals), "focus": focus}


def condition_mean_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for path in sorted(root.glob("*.json")):
        obj = load_json(path)
        groups = obj.get("groups") or {}
        rows.append(
            {
                "path": str(path),
                "name": path.name,
                "groups": {
                    str(group): len((payload or {}).get("condition_metrics") or [])
                    for group, payload in groups.items()
                },
            }
        )
    return rows


def find_support_set_task_artifacts() -> list[str]:
    hits: list[str] = []
    for root in (ROOT / "reports", ROOT / "runs"):
        if not root.exists():
            continue
        for path in root.rglob("*support_set_task*"):
            if path.is_file() and path.name not in {OUT_JSON.name, OUT_MD.name}:
                hits.append(str(path))
    return sorted(hits)


def main() -> int:
    protocol = load_json(PROTOCOL_JSON)
    code = load_json(CODE_JSON)
    split = load_json(SPLIT)
    counts = split_counts(split)
    inventory = condition_mean_inventory(OLD_SUPPORT_ROOT)
    support_set_task_artifacts = find_support_set_task_artifacts()

    train_multi_condition_means = [
        item
        for item in inventory
        if any("train_multi" == group for group in item.get("groups", {}))
    ]
    support_val_condition_means = [
        item
        for item in inventory
        if any(group in {"test", "test_multi", "support_val_multi"} for group in item.get("groups", {}))
    ]

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, evidence: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    add(
        "protocol_ready",
        protocol.get("status") == "trackc_support_set_task_adapter_protocol_ready_code_gate_next_no_gpu",
        f"status={protocol.get('status')}",
    )
    add(
        "code_boundary_pass",
        code.get("status") == "trackc_support_set_task_adapter_code_boundary_pass_cpu_metric_gate_next_no_gpu"
        and not code.get("failed_checks"),
        f"status={code.get('status')} failed_checks={len(code.get('failed_checks') or [])}",
    )
    add(
        "safe_split_has_train_and_support_multi",
        counts["totals"].get("train_multi", 0) > 0 and counts["totals"].get("support_val_multi", 0) > 0,
        f"totals={counts['totals']} focus={counts['focus']}",
    )
    add(
        "train_multi_condition_means_exist",
        bool(train_multi_condition_means),
        f"train_multi condition-mean artifacts found={len(train_multi_condition_means)}",
    )
    add(
        "support_val_condition_means_exist",
        bool(support_val_condition_means),
        f"support/test condition-mean artifacts found={len(support_val_condition_means)}",
    )
    add(
        "distinct_support_set_task_metric_artifact_exists",
        any(
            "cpu_metric_gate" not in path
            and "code_boundary" not in path
            and "protocol" not in path
            for path in support_set_task_artifacts
        ),
        f"support_set_task artifacts={support_set_task_artifacts}",
    )
    add(
        "old_support_context_artifacts_not_accepted_as_new_gate",
        not train_multi_condition_means and bool(support_val_condition_means),
        "existing condition means are support/canonical only from consumed support-Film residual blend; do not relabel them as support-set task adapter evidence",
    )

    failed = [row for row in checks if not row["passed"]]
    status = (
        "trackc_support_set_task_adapter_cpu_metric_gate_pass_launcher_gate_next_no_gpu"
        if not failed
        else "trackc_support_set_task_adapter_cpu_metric_gate_fail_no_gpu_missing_train_multi_task_inputs"
    )
    payload = {
        "status": status,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpu_authorization": "none",
        "next_authorization": "none" if failed else "launcher_provenance_gate_only",
        "checks": checks,
        "failed_checks": failed,
        "split": {
            "path": str(SPLIT),
            "sha256": sha256(SPLIT),
            "counts": counts,
        },
        "condition_mean_inventory": inventory,
        "support_set_task_artifacts": support_set_task_artifacts,
        "decision": {
            "close_or_keep": "keep_protocol_but_do_not_launch_gpu",
            "reason": (
                "The code boundary exists, and the safe split has train/support multi rows, "
                "but no train_multi condition-mean/task-summary metric artifact exists for "
                "the distinct support-set task adapter. Reusing old support-Film residual "
                "blend artifacts would violate the protocol distinction."
            ),
            "valid_next_action": (
                "Design an input-generation gate for train_multi support-set task summaries "
                "or pivot to another query-free CPU gate; do not launch GPU from this evidence."
            ),
        },
        "forbidden": [
            "do not use full v2 held-out query",
            "do not use canonical test_multi for selection",
            "do not treat consumed support_context/residual/FiLM artifacts as support-set task adapter metrics",
            "do not launch GPU from this failed input gate",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Track C Support-Set Task Adapter CPU Metric Gate",
        "",
        f"Status: `{status}`",
        "GPU authorization: `none`",
        f"Next authorization: `{payload['next_authorization']}`",
        "",
        "## Decision",
        "",
        payload["decision"]["reason"],
        "",
        "## Split Counts",
        "",
        f"- safe split: `{SPLIT}`",
        f"- totals: `{counts['totals']}`",
        f"- focus: `{counts['focus']}`",
        "",
        "## Checks",
        "",
        "| check | passed | evidence |",
        "|---|---:|---|",
    ]
    for row in checks:
        evidence = str(row["evidence"]).replace("\n", " ")
        lines.append(f"| `{row['name']}` | `{row['passed']}` | {evidence} |")
    lines.extend(["", "## Condition-Mean Inventory", ""])
    for item in inventory:
        lines.append(f"- `{item['name']}` groups `{item['groups']}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This failed gate does not close the support-set task adapter architecture.",
            "- It blocks GPU launch until a real train_multi support-set task metric/input artifact exists.",
            "- Existing support-Film residual blend artifacts remain historical evidence only.",
            "",
            "## Valid Next Action",
            "",
            payload["decision"]["valid_next_action"],
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))
    print(json.dumps({"status": status, "failed": len(failed), "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
