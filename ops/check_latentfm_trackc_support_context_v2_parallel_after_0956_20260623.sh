#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
WINDOW_EPOCH=$(TZ=Asia/Shanghai date -d '2026-06-23 09:56:00' +%s)
NOW_EPOCH=$(TZ=Asia/Shanghai date +%s)
if (( NOW_EPOCH < WINDOW_EPOCH )); then
  echo "Refusing to check before 2026-06-23 09:56:00 CST; v2 parallel variants are long GPU tasks." >&2
  exit 3
fi

PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

"${PYTHON}" - <<'PY'
import json
import sys
from pathlib import Path

root = Path("/data/cyx/1030/scLatent")
runs = [
    "xverse_trackc_support_context_v2_residual_ep050_replay2_2k_seed42",
    "xverse_trackc_support_context_v2_contextc_ep050_replay2_2k_seed42",
]
run_base = root / "runs/latentfm_xverse_trackc_support_context_v2_parallel_20260623"
reports = root / "reports"
allowed_pass = "trackc_smoke_support_pass_needs_uncapped_noharm_before_query"
rows = []
overall = 0
for run in runs:
    run_root = run_base / run
    train_exit_path = run_root / f"{run}.EXIT_CODE"
    posthoc_exit_path = run_root / f"{run}.POSTHOC_EXIT_CODE"
    decision_json = reports / f"latentfm_trackc_routed_distill_smoke_decision_{run}.json"
    decision_md = reports / f"LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_{run}.md"
    row = {"run": run, "run_root": str(run_root)}
    if not train_exit_path.exists():
        row["status"] = "training_or_exit_pending"
        overall = max(overall, 3)
        rows.append(row)
        continue
    train_exit = train_exit_path.read_text(encoding="utf-8").strip()
    row["training_exit"] = train_exit
    if train_exit != "0":
        row["status"] = "training_failed"
        overall = max(overall, 2)
        rows.append(row)
        continue
    if not posthoc_exit_path.exists():
        row["status"] = "posthoc_pending"
        overall = max(overall, 3)
        rows.append(row)
        continue
    posthoc_exit = posthoc_exit_path.read_text(encoding="utf-8").strip()
    row["posthoc_exit"] = posthoc_exit
    if posthoc_exit != "0":
        row["status"] = "posthoc_failed"
        overall = max(overall, 2)
        rows.append(row)
        continue
    if not decision_json.exists() or not decision_md.exists():
        row["status"] = "decision_missing"
        overall = max(overall, 2)
        rows.append(row)
        continue
    payload = json.loads(decision_json.read_text(encoding="utf-8"))
    decision_status = payload.get("status") or (payload.get("decision") or {}).get("status")
    row["decision_status"] = decision_status
    row["decision_md"] = str(decision_md)
    support_absent_ok = True
    for rel in [
        "posthoc_eval/canonical_candidate_split_ode20_stablecaps.json",
        "posthoc_eval/canonical_candidate_family_ode20_stablecaps.json",
    ]:
        p = run_root / rel
        if not p.exists():
            support_absent_ok = False
            row.setdefault("missing_canonical_json", []).append(str(p))
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        if j.get("support_context_forced_absent") is not True:
            support_absent_ok = False
            row.setdefault("bad_support_absent_json", []).append(str(p))
    row["canonical_support_context_forced_absent"] = support_absent_ok
    if not support_absent_ok:
        row["status"] = "canonical_support_absent_provenance_fail"
        overall = max(overall, 2)
    elif decision_status == allowed_pass:
        row["status"] = "pass_needs_uncapped_noharm"
    else:
        row["status"] = "terminal_or_closed"
    rows.append(row)

print(json.dumps({"status": "checked", "rows": rows}, indent=2))
raise SystemExit(overall)
PY
