#!/usr/bin/env python3
"""
GPU queue: each of ``--gpus`` runs one model at a time (all datasets serially), then
pulls the next model from a shared queue until empty.

Uses one subprocess per dataset via export_embedding_one.py for clean memory.
Each model runs in its own Python interpreter (see model_registry.python_for_model).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, TextIO

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model_registry import (
    DEFAULT_EMBEDDING_EXPORT_ROOT,
    DEFAULT_EMBEDDING_LOGS_DIR,
    DEFAULT_EMBEDDING_RUNS_DIR,
    LATENT_BENCH_ROOT,
    MODEL_QUEUE_ORDER,
    check_weights,
    python_for_model,
    subprocess_env,
)
import paths

EXPORT_SCRIPT = _ROOT / "export_embedding_one.py"


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_queue_from_preflight(preflight: Path) -> List[str]:
    obj = json.loads(preflight.read_text())
    return list(obj.get("queue_order") or [])


def thread_gpu_env(gpu_id: str) -> Dict[str, str]:
    e = os.environ.copy()
    e["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    e["OMP_NUM_THREADS"] = "2"
    e["MKL_NUM_THREADS"] = "2"
    e["OPENBLAS_NUM_THREADS"] = "2"
    e["NUMEXPR_NUM_THREADS"] = "2"
    e["TORCHINDUCTOR_COMPILE_THREADS"] = "2"
    e["MAX_JOBS"] = "2"
    e["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cache = paths.cache_root()
    e.setdefault("HF_HOME", str(cache / "huggingface"))
    e.setdefault("TRANSFORMERS_CACHE", str(cache / "huggingface" / "transformers"))
    e.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    e.setdefault("TORCH_HOME", str(cache / "torch"))
    return e


def _fmt_eta(sec: float) -> str:
    if sec < 0 or sec != sec or sec == float("inf"):
        return "?"
    s = int(sec)
    h, r = divmod(s, 3600)
    m, s2 = divmod(r, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s2}s"
    return f"{s2}s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True, help="manifest.jsonl from preflight_embedding")
    ap.add_argument("--preflight", type=Path, default=None, help="preflight.json; if set, default model queue from queue_order")
    ap.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_EMBEDDING_EXPORT_ROOT,
        help="Default: LATENT_BENCH_OUTPUT_ROOT/embeddings (scFM output/embeddings)",
    )
    ap.add_argument(
        "--status-jsonl",
        type=Path,
        default=DEFAULT_EMBEDDING_RUNS_DIR / "run_status.jsonl",
    )
    ap.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_EMBEDDING_LOGS_DIR / "embedding_queue.log",
        help="Append human-readable progress, subprocess stdout/stderr excerpts",
    )
    ap.add_argument("--gpus", type=str, nargs="+", default=[str(i) for i in range(4)])
    ap.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Override model order; default from preflight or MODEL_QUEUE_ORDER",
    )
    ap.add_argument("--batch-size", type=int, default=8, help="Conservative default for GPU memory")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-cells", type=int, default=0, help="0=all; dry-run use 256")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="sets max-cells=256 if max-cells=0")
    ap.add_argument(
        "--abort-after-consecutive-fails",
        type=int,
        default=0,
        help="If >0, stop running further datasets for the current model on this GPU after "
        "this many consecutive subprocess failures (then take next model from queue). 0=never.",
    )
    args = ap.parse_args()

    if args.dry_run and args.max_cells == 0:
        args.max_cells = 256

    args.manifest = args.manifest.resolve()
    if args.preflight is not None:
        args.preflight = args.preflight.resolve()
    args.export_root = args.export_root.resolve()
    args.status_jsonl = args.status_jsonl.resolve()
    args.log_file = args.log_file.resolve()

    manifest = load_manifest(args.manifest)
    if not manifest:
        print("empty manifest", file=sys.stderr)
        return 1

    if args.models:
        model_list = [m.lower().strip() for m in args.models]
    elif args.preflight and args.preflight.is_file():
        model_list = load_queue_from_preflight(args.preflight)
    else:
        model_list = list(MODEL_QUEUE_ORDER)

    if not args.dry_run:
        missing = []
        for m in model_list:
            status, detail = check_weights(m)
            if status != "ready":
                missing.append(f"{m}: {status} {detail}")
        if missing:
            print("Resource validation failed before live embedding queue:", file=sys.stderr)
            for row in missing:
                print(f"  - {row}", file=sys.stderr)
            print(paths.describe_layout(), file=sys.stderr)
            return 2

    n_ds = len(manifest)
    total_export_tasks = len(model_list) * n_ds
    started_wall = time.time()

    args.export_root.mkdir(parents=True, exist_ok=True)
    args.status_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    model_q: queue.Queue[str] = queue.Queue()
    for m in model_list:
        model_q.put(m)

    lock = threading.Lock()
    done_tasks: List[int] = [0]  # mutable counter for global progress
    recent_sec: Deque[float] = deque(maxlen=48)

    log_fp: Optional[TextIO] = open(args.log_file, "a", encoding="utf-8")

    def log_line(msg: str, *, also_print: bool = True) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{ts}] {msg}\n"
        if also_print:
            print(line, end="", flush=True)
        if log_fp:
            log_fp.write(line)
            log_fp.flush()

    def log_status(obj: Dict[str, Any]) -> None:
        obj["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with lock:
            with open(args.status_jsonl, "a") as f:
                f.write(json.dumps(obj, default=str) + "\n")

    def bump_done(dt: float) -> None:
        with lock:
            done_tasks[0] += 1
            recent_sec.append(dt)

    def global_eta_line() -> str:
        with lock:
            d = done_tasks[0]
            if d <= 0 or not recent_sec:
                return f"global {d}/{total_export_tasks}"
            avg = sum(recent_sec) / len(recent_sec)
            rem = max(0, total_export_tasks - d)
            return f"global {d}/{total_export_tasks} (~{100.0 * d / total_export_tasks:.1f}%) ETA~{_fmt_eta(rem * avg)} (avg {_fmt_eta(avg)}/task)"

    try:
        log_line(
            f"=== embedding queue start === manifest={args.manifest} models={len(model_list)} datasets={n_ds} "
            f"total_tasks={total_export_tasks} gpus={args.gpus} batch_size={args.batch_size} export_root={args.export_root}",
        )

        def worker(gpu_slot: str) -> None:
            while True:
                try:
                    model = model_q.get_nowait()
                except queue.Empty:
                    return
                py = python_for_model(model)
                log_line(f"GPU{gpu_slot} START_MODEL model={model} python={py} (env for this model)", also_print=True)
                consec_fail = 0
                ds_idx = 0
                for row in manifest:
                    ds_idx += 1
                    adata_path = Path(row["path"]).resolve()
                    ds_id = row.get("dataset_id") or adata_path.stem
                    out_dir = args.export_root / model / ds_id / "raw"
                    base_env = subprocess_env(model, thread_gpu_env(gpu_slot))
                    cmd = [
                        py,
                        str(EXPORT_SCRIPT),
                        "--model",
                        model,
                        "--adata",
                        str(adata_path),
                        "--out-dir",
                        str(out_dir),
                        "--device",
                        args.device,
                        "--batch-size",
                        str(args.batch_size),
                    ]
                    if args.max_cells:
                        cmd.extend(["--max-cells", str(args.max_cells)])
                    if args.skip_existing:
                        cmd.append("--skip-existing")
                    if args.dry_run:
                        log_line(
                            "DRY_RUN "
                            + " ".join(
                                [
                                    f"CUDA_VISIBLE_DEVICES={gpu_slot}",
                                    py,
                                    str(EXPORT_SCRIPT),
                                    "--model",
                                    model,
                                    "--adata",
                                    str(adata_path),
                                    "--out-dir",
                                    str(out_dir),
                                    "--device",
                                    args.device,
                                    "--batch-size",
                                    str(args.batch_size),
                                    *(["--max-cells", str(args.max_cells)] if args.max_cells else []),
                                    *(["--skip-existing"] if args.skip_existing else []),
                                ]
                            )
                        )
                        bump_done(0.0)
                        continue
                    rec = {
                        "event": "start",
                        "gpu": gpu_slot,
                        "model": model,
                        "dataset_id": ds_id,
                        "adata": str(adata_path),
                        "python": py,
                        "dataset_index": ds_idx,
                        "datasets_total": n_ds,
                    }
                    log_status(rec)
                    log_line(
                        f"GPU{gpu_slot} [{ds_idx}/{n_ds}] START model={model} dataset={ds_id} | {global_eta_line()}",
                    )
                    t0 = time.time()
                    r = subprocess.run(
                        cmd,
                        cwd=str(LATENT_BENCH_ROOT),
                        env=base_env,
                        capture_output=True,
                        text=True,
                    )
                    elapsed = time.time() - t0
                    ok = r.returncode == 0
                    if ok:
                        consec_fail = 0
                    else:
                        consec_fail += 1
                    bump_done(elapsed)
                    tail_out = (r.stdout or "")[-4000:]
                    tail_err = (r.stderr or "")[-8000:]
                    log_line(
                        f"GPU{gpu_slot} [{ds_idx}/{n_ds}] {'DONE' if ok else 'FAIL'} model={model} dataset={ds_id} "
                        f"rc={r.returncode} wall={elapsed:.1f}s | {global_eta_line()} | consec_fail={consec_fail}",
                    )
                    if tail_out.strip():
                        log_line(f"GPU{gpu_slot} stdout (tail):\n{tail_out}", also_print=False)
                    if tail_err.strip():
                        log_line(f"GPU{gpu_slot} stderr (tail):\n{tail_err}", also_print=False)

                    rec2 = {
                        "event": "done" if ok else "failed",
                        "gpu": gpu_slot,
                        "model": model,
                        "dataset_id": ds_id,
                        "returncode": r.returncode,
                        "seconds": round(elapsed, 2),
                        "python": py,
                    }
                    log_status(rec2)

                    if (
                        not ok
                        and args.abort_after_consecutive_fails
                        and consec_fail >= args.abort_after_consecutive_fails
                    ):
                        log_line(
                            f"GPU{gpu_slot} SKIP_MODEL remaining datasets for model={model} "
                            f"after {consec_fail} consecutive failures (abort threshold={args.abort_after_consecutive_fails})",
                        )
                        log_status(
                            {
                                "event": "model_aborted",
                                "gpu": gpu_slot,
                                "model": model,
                                "reason": "consecutive_failures",
                                "consecutive_failures": consec_fail,
                            }
                        )
                        break

                log_line(f"GPU{gpu_slot} FINISH_MODEL model={model}", also_print=True)

        threads = []
        for g in args.gpus:
            th = threading.Thread(target=worker, args=(g,), daemon=True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

        total_wall = time.time() - started_wall
        log_line(f"=== embedding queue finished wall_time={_fmt_eta(total_wall)} log={args.log_file} ===")
    finally:
        if log_fp:
            log_fp.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
