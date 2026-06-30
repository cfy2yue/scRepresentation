#!/usr/bin/env python3
"""Audit recent LatentFM RUN_STATUS files against EXIT_CODE and tmux state."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_run_status_consistency_audit_20260623.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_RUN_STATUS_CONSISTENCY_AUDIT_20260623.md"


@dataclass(frozen=True)
class RunStatus:
    path: Path
    mtime: float
    session: str | None
    status_text: str
    exit_code: str | None
    finished: str | None
    tmux_present: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def tmux_sessions() -> set[str]:
    proc = subprocess.run(["tmux", "ls"], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return set()
    sessions = set()
    for line in proc.stdout.splitlines():
        if ":" in line:
            sessions.add(line.split(":", 1)[0].strip())
    return sessions


def extract_session(text: str) -> str | None:
    marker = "tmux:"
    for line in text.splitlines():
        if marker in line and "`" in line:
            parts = line.split("`")
            if len(parts) >= 2:
                return parts[1].strip()
    return None


def extract_status(text: str) -> str:
    marker = "## Current status"
    if marker not in text:
        return ""
    tail = text.split(marker, 1)[1].strip()
    if not tail:
        return ""
    lines = []
    for line in tail.splitlines():
        if line.startswith("## ") and lines:
            break
        if line.strip():
            lines.append(line.strip())
        if len(lines) >= 4:
            break
    return " ".join(lines)


def read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def collect(limit: int) -> list[RunStatus]:
    sessions = tmux_sessions()
    files = sorted((ROOT / "runs").glob("**/RUN_STATUS.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for path in files[:limit]:
        text = path.read_text(encoding="utf-8", errors="replace")
        session = extract_session(text)
        rows.append(
            RunStatus(
                path=path,
                mtime=path.stat().st_mtime,
                session=session,
                status_text=extract_status(text),
                exit_code=read_optional(path.parent / "EXIT_CODE"),
                finished=read_optional(path.parent / "FINISHED"),
                tmux_present=bool(session and session in sessions),
            )
        )
    return rows


def classify(row: RunStatus) -> tuple[str, str]:
    status = row.status_text.lower()
    if row.exit_code is not None and "started" in status:
        return "mismatch", "EXIT_CODE exists but status still says Started"
    if row.exit_code is None and not row.tmux_present and "finished" not in status and "failed" not in status:
        return "needs_review", "no EXIT_CODE and tmux session absent"
    if row.exit_code is not None and row.exit_code.strip() != "0" and "failed" not in status and "superseded" not in status:
        return "needs_review", "nonzero EXIT_CODE not clearly marked failed/superseded"
    return "ok", "consistent"


def write_outputs(rows: list[RunStatus], out_json: Path, out_md: Path) -> None:
    payload = []
    counts: dict[str, int] = {}
    for row in rows:
        label, reason = classify(row)
        counts[label] = counts.get(label, 0) + 1
        payload.append(
            {
                "path": str(row.path),
                "session": row.session,
                "status_text": row.status_text,
                "exit_code": row.exit_code,
                "finished": row.finished,
                "tmux_present": row.tmux_present,
                "audit_status": label,
                "reason": reason,
            }
        )
    out_json.write_text(json.dumps({"status_counts": counts, "rows": payload}, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM RUN_STATUS Consistency Audit",
        "",
        f"Generated: `{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}`",
        "",
        "Scope: latest RUN_STATUS files under `/data/cyx/1030/scLatent/runs`. This audit is read-only with respect to experiments: no GPU, no data evaluation, no query access.",
        "",
        "## Summary",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    for key in sorted(counts):
        lines.append(f"| `{key}` | {counts[key]} |")
    lines.extend(["", "## Rows", "", "| audit | reason | exit | tmux | status text | path |", "|---|---|---:|---|---|---|"])
    for item in payload:
        path = item["path"]
        rel = str(Path(path).relative_to(ROOT))
        lines.append(
            "| `{audit_status}` | {reason} | `{exit_code}` | `{tmux}` | {status} | `{path}` |".format(
                audit_status=item["audit_status"],
                reason=item["reason"],
                exit_code=item["exit_code"],
                tmux="present" if item["tmux_present"] else "absent",
                status=(item["status_text"] or "").replace("|", "\\|"),
                path=rel,
            )
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = collect(args.limit)
    write_outputs(rows, args.out_json, args.out_md)
    bad = [row for row in rows if classify(row)[0] != "ok"]
    print(json.dumps({"status": "run_status_consistency_audit_complete", "non_ok": len(bad), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
