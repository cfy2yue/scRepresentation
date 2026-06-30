#!/usr/bin/env python3
"""
Execute ``run_metrics_one.py`` for each line of ``run_manifest.jsonl``.

Appends JSON lines to ``output/metrics/run_status.jsonl`` (started / done / failed).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

BENCH_ROOT = Path(__file__).resolve().parents[1]
SCFM_ROOT = BENCH_ROOT.parent
FM_ROOT = SCFM_ROOT / "fm"
sys.path.insert(0, str(FM_ROOT))
import paths

_METRIC_JSON = ("atlas.json", "geometry.json", "perturb.json", "summary.json")


def _pick_python(explicit: Path | None) -> str:
    if explicit is not None:
        return str(explicit)
    return os.environ.get("SCFM_METRICS_PYTHON", "python3")


def _migrate_legacy_layout(scfm: Path) -> List[str]:
    """
    Move legacy ``output/metrics/<m>/<d>/*.json`` into ``.../<m>/<d>/raw/`` (idempotent).
    """
    metrics_root = paths.output_root() / "metrics"
    if not metrics_root.is_dir():
        return []
    migrated: List[str] = []
    for model_dir in sorted(metrics_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.endswith(".jsonl") or model_dir.name.endswith(".csv"):
            continue
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            top_summ = ds_dir / "summary.json"
            raw_dir = ds_dir / "raw"
            raw_summ = raw_dir / "summary.json"
            if not top_summ.is_file() or raw_summ.is_file():
                continue
            raw_dir.mkdir(parents=True, exist_ok=True)
            for name in _METRIC_JSON:
                p = ds_dir / name
                if p.is_file():
                    dest = raw_dir / name
                    if not dest.exists():
                        p.rename(dest)
                        migrated.append(str(dest))
    if migrated:
        print("legacy_metrics_migrated", json.dumps({"n_files": len(migrated), "sample": migrated[:8]}))
    return migrated


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scfm-root", type=Path, default=SCFM_ROOT)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Default: SCFM_OUTPUT_ROOT/metrics/run_manifest.jsonl",
    )
    ap.add_argument(
        "--status-log",
        type=Path,
        default=None,
        help="Default: SCFM_OUTPUT_ROOT/metrics/run_status.jsonl",
    )
    ap.add_argument("--python", type=Path, default=None, help="Interpreter for run_metrics_one")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows whose summary.json already exists for this latent_space",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max tasks (0 = all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--latent-space",
        choices=("raw", "pca128", "both"),
        default="both",
        help="Metric latent space; both runs raw then pca128 per manifest row",
    )
    ap.add_argument(
        "--no-migrate",
        action="store_true",
        help="Skip one-time legacy layout migration (json at metrics/<m>/<d>/ → raw/)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (each spawns one run_metrics_one.py at a time).",
    )
    args = ap.parse_args()

    scfm = args.scfm_root.resolve()
    if not args.no_migrate:
        _migrate_legacy_layout(scfm)

    manifest = args.manifest or (paths.output_root() / "metrics" / "run_manifest.jsonl")
    status_log = args.status_log or (paths.output_root() / "metrics" / "run_status.jsonl")
    status_log.parent.mkdir(parents=True, exist_ok=True)

    py = _pick_python(args.python)
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{BENCH_ROOT}:{FM_ROOT}" + (f":{prev}" if prev else "")

    spaces: List[str]
    if args.latent_space == "both":
        spaces = ["raw", "pca128"]
    else:
        spaces = [args.latent_space]

    rows: List[Dict[str, Any]] = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    tasks: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if args.limit and i >= args.limit:
            break
        base_out = Path(row["out_dir"])
        for latent_space in spaces:
            out_dir = base_out / latent_space
            summ = out_dir / "summary.json"
            if args.resume and summ.is_file():
                tasks.append({"_skip": True, "latent_space": latent_space, "row": row})
                continue
            cmd = [py, str(BENCH_ROOT / "cli" / "run_metrics_one.py")]
            argv: List[str] = row.get("argv") or []
            if argv and argv[0].endswith("run_metrics_one.py"):
                extra = argv[1:]
            else:
                extra = argv
            cmd.extend(extra)
            cmd.extend(["--latent-space", latent_space])
            tasks.append({"_skip": False, "latent_space": latent_space, "row": row, "cmd": cmd})

    def _run_task(task: Dict[str, Any]) -> Dict[str, Any]:
        row = task["row"]
        latent_space = task["latent_space"]
        ev_base = {
            "model": row["model"],
            "dataset_id": row["dataset_id"],
            "category": row.get("category"),
            "latent_space": latent_space,
        }
        if task["_skip"]:
            return {"event": "skip", **ev_base, "returncode": 0, "seconds": 0.0}
        cmd = task["cmd"]
        with open(status_log, "a") as slog:
            slog.write(
                json.dumps(
                    {"event": "start", **ev_base, "cmd": cmd,
                     "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                    default=str,
                )
                + "\n"
            )
        if args.dry_run:
            return {"event": "done", **ev_base, "returncode": 0, "seconds": 0.0}
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(scfm), env=env, capture_output=True, text=True)
        dt = time.time() - t0
        rec: Dict[str, Any] = {
            "event": "done" if proc.returncode == 0 else "failed",
            **ev_base,
            "returncode": proc.returncode,
            "seconds": round(dt, 3),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if proc.returncode != 0:
            rec["stderr_tail"] = (proc.stderr or "")[-4000:]
        with open(status_log, "a") as slog:
            slog.write(json.dumps(rec, default=str) + "\n")
        return rec

    done_n = skip_n = fail_n = 0
    workers = max(1, int(args.workers))
    if workers == 1:
        for t in tasks:
            r = _run_task(t)
            ev = r.get("event")
            if ev == "skip":
                skip_n += 1
            elif ev == "failed":
                fail_n += 1
            else:
                done_n += 1
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_task, t) for t in tasks]
            for fut in as_completed(futs):
                r = fut.result()
                ev = r.get("event")
                if ev == "skip":
                    skip_n += 1
                elif ev == "failed":
                    fail_n += 1
                else:
                    done_n += 1

    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "latent_spaces": spaces,
                "ran_ok": done_n,
                "resume_skipped": skip_n,
                "failed": fail_n,
            }
        )
    )
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
