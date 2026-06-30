#!/usr/bin/env python3
"""Summarize output/embedding_runs/run_status.jsonl into done/failed counts and retry manifest lines."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model_registry import DEFAULT_EMBEDDING_RUNS_DIR


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--status-jsonl",
        type=Path,
        default=DEFAULT_EMBEDDING_RUNS_DIR / "run_status.jsonl",
    )
    ap.add_argument("--out-json", type=Path, default=None, help="Full summary JSON")
    ap.add_argument(
        "--retry-manifest",
        type=Path,
        default=None,
        help="Write manifest.jsonl lines only for (model,dataset) with failed done event",
    )
    args = ap.parse_args()

    if not args.status_jsonl.is_file():
        print(json.dumps({"error": f"missing {args.status_jsonl}"}), file=sys.stderr)
        return 1

    events = load_jsonl(args.status_jsonl)
    by_key: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for ev in events:
        m = ev.get("model")
        ds = ev.get("dataset_id")
        if m and ds:
            by_key[(str(m), str(ds))].append(ev)

    done_ok: List[Dict[str, str]] = []
    failed: List[Dict[str, Any]] = []
    incomplete: List[Tuple[str, str]] = []

    for (model, ds_id), evs in sorted(by_key.items()):
        events_sorted = sorted(evs, key=lambda e: e.get("ts", ""))
        last_done = None
        for e in events_sorted:
            if e.get("event") == "done" and e.get("returncode") == 0:
                last_done = e
            elif e.get("event") == "failed":
                last_done = e
        if last_done is None:
            incomplete.append((model, ds_id))
            continue
        if last_done.get("event") == "done" and last_done.get("returncode") == 0:
            done_ok.append({"model": model, "dataset_id": ds_id})
        else:
            failed.append(
                {
                    "model": model,
                    "dataset_id": ds_id,
                    "returncode": last_done.get("returncode"),
                    "gpu": last_done.get("gpu"),
                    "seconds": last_done.get("seconds"),
                }
            )

    summary = {
        "status_jsonl": str(args.status_jsonl),
        "n_events": len(events),
        "n_pairs": len(by_key),
        "done_ok": len(done_ok),
        "failed": len(failed),
        "incomplete": len(incomplete),
        "failed_detail": failed,
        "incomplete_pairs": [{"model": a, "dataset_id": b} for a, b in incomplete],
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, default=str))

    if args.retry_manifest:
        # Need adata paths: infer from last "start" event per failed pair
        adata_by_pair: Dict[Tuple[str, str], str] = {}
        for ev in events:
            if ev.get("event") != "start":
                continue
            key = (str(ev.get("model")), str(ev.get("dataset_id")))
            if ev.get("adata"):
                adata_by_pair[key] = str(ev["adata"])

        lines: List[str] = []
        seen_ds: Dict[str, str] = {}
        for ev in events:
            if ev.get("event") == "start" and ev.get("adata"):
                did = str(ev.get("dataset_id"))
                seen_ds[did] = str(ev["adata"])

        with open(args.retry_manifest, "w") as mf:
            for item in failed:
                m, d = item["model"], item["dataset_id"]
                apath = adata_by_pair.get((m, d)) or seen_ds.get(d)
                if not apath:
                    continue
                mf.write(
                    json.dumps(
                        {"path": apath, "dataset_id": d, "category": "retry"},
                        default=str,
                    )
                    + "\n"
                )

    print(json.dumps({"done_ok": summary["done_ok"], "failed": summary["failed"], "incomplete": summary["incomplete"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
