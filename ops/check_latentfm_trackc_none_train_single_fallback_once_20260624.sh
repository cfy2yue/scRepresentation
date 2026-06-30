#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_ROOT=${ROOT}/runs/latentfm_trackc_support_only_robustness_20260624
SUMMARY=${ROOT}/ops/summarize_latentfm_trackc_support_only_pairtype_strata_20260624.py
OUT_JSON=${ROOT}/reports/latentfm_trackc_none_train_single_fallback_stability_20260624.json
OUT_MD=${ROOT}/reports/LATENTFM_TRACKC_NONE_TRAIN_SINGLE_FALLBACK_STABILITY_20260624.md
RUNS=(
  xverse_trackc_support_pairtype_none_train_single_resfilm_ep050_replay2_2k_seed43
  xverse_trackc_support_pairtype_none_train_single_resfilm_ep050_replay2_2k_seed44
)

for run in "${RUNS[@]}"; do
  run_dir=${RUN_ROOT}/${run}
  if [[ ! -f "${run_dir}/${run}.EXIT_CODE" ]]; then
    echo "pending: training not complete for ${run}"
    exit 0
  fi
  if [[ "$(cat "${run_dir}/${run}.EXIT_CODE")" != "0" ]]; then
    echo "failed: training exit $(cat "${run_dir}/${run}.EXIT_CODE") for ${run}" >&2
    exit 3
  fi
  if [[ ! -f "${run_dir}/POSTHOC_EXIT_CODE" ]]; then
    echo "pending: posthoc not complete for ${run}"
    exit 0
  fi
  if [[ "$(cat "${run_dir}/POSTHOC_EXIT_CODE")" != "0" ]]; then
    echo "failed: posthoc exit $(cat "${run_dir}/POSTHOC_EXIT_CODE") for ${run}" >&2
    exit 3
  fi
done

tmp_json=${ROOT}/reports/latentfm_trackc_support_only_pairtype_strata_summary_20260624_none_train_single.json
tmp_md=${ROOT}/reports/LATENTFM_TRACKC_SUPPORT_ONLY_PAIRTYPE_STRATA_SUMMARY_20260624_none_train_single.md
"${PYTHON}" "${SUMMARY}" --target-label none_train_single --runs "${RUNS[@]}" >/tmp/latentfm_none_train_single_fallback_summary_$$.log

"${PYTHON}" - "${tmp_json}" "${OUT_JSON}" "${OUT_MD}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
out_json = Path(sys.argv[2])
out_md = Path(sys.argv[3])
run_root = Path(sys.argv[4])
payload = json.loads(src.read_text(encoding="utf-8"))
runs = payload.get("runs") or []
completed = [r for r in runs if r.get("status") != "incomplete"]
passed = [r for r in completed if r.get("status") == "pass_pairtype_target_support_control"]
failed = [r for r in completed if r.get("status", "").startswith("fail_")]
whole_support = []
whole_support_fail = []
for row in runs:
    run_name = str(row.get("run_name") or "")
    if not run_name:
        continue
    path = Path("/data/cyx/1030/scLatent/reports") / f"latentfm_trackc_support_only_robustness_decision_{run_name}.json"
    if not path.exists():
        whole_support.append({"run_name": run_name, "status": "missing", "path": str(path)})
        whole_support_fail.append(run_name)
        continue
    obj = json.loads(path.read_text(encoding="utf-8"))
    status0 = (obj.get("decision") or {}).get("status") or obj.get("status")
    reasons = (obj.get("decision") or {}).get("reasons") or obj.get("reasons") or []
    whole_support.append({"run_name": run_name, "status": status0, "reasons": reasons, "path": str(path)})
    if status0 != "trackc_support_only_robustness_pass_support_gate":
        whole_support_fail.append(run_name)
if len(completed) < 2:
    status = "pending"
    action = "wait for both seed43 and seed44 posthoc"
elif len(passed) == 2 and not failed and not whole_support_fail:
    status = "pass_two_seed_support_control_external_review_next"
    action = "request external review; no canonical no-harm/query yet"
elif len(passed) == 2 and not failed:
    status = "target_stratum_signal_only_no_promotion"
    action = "close or relabel none_train_single fallback; whole-support robustness failed, so no canonical no-harm/query"
else:
    status = "fail_close_none_train_single_fallback"
    action = "close none_train_single fallback; do not launch canonical no-harm/query"

decision = {
    "status": status,
    "action": action,
    "boundary": payload.get("boundary"),
    "source_summary_json": str(src),
    "rules": {
        "two_seed_gate": "seed43 and seed44 must both pass support-control with no failed seed",
        "canonical_noharm_allowed": False,
        "heldout_query_allowed": False,
        "whole_support_gate": "all same-run robustness decisions must pass before any no-harm wrapper; target-stratum pass alone is non-promotional",
    },
    "runs": runs,
    "whole_support_decisions": whole_support,
    "n_completed": len(completed),
    "n_pass": len(passed),
    "failed_runs": [r.get("run_name") for r in failed],
    "whole_support_failed_runs": whole_support_fail,
}
out_json.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = [
    "# Track C None-Train-Single Fallback Stability",
    "",
    f"Status: `{status}`",
    f"Action: `{action}`",
    "",
    "## Boundary",
    "",
    "- Query-free and canonical-free.",
    "- Reads only safe trainselect support-val posthoc/control JSONs.",
    "- Two-seed target-stratum gate for seed43/44 only.",
    "- Target-stratum pass is non-promotional unless the corresponding whole-support robustness decisions also pass.",
    "- Canonical no-harm/query remains forbidden by this checker.",
    "",
    "## Seed Rows",
    "",
    "| run | status | target pp | target MMD | zero pp | shuffle pp | absent pp | reasons |",
    "|---|---|---:|---:|---:|---:|---:|---|",
]
for row in runs:
    target = row.get("target") or {}
    def val(name):
        obj = target.get(name) or {}
        x = obj.get("equal_dataset_mean_delta")
        return "NA" if x is None else f"{float(x):+.6f}"
    lines.append(
        f"| `{row.get('run_name')}` | `{row.get('status')}` | "
        f"{val('actual_pp')} | {val('actual_mmd')} | {val('zero_pp')} | "
        f"{val('shuffle_pp')} | {val('absent_pp')} | `{row.get('reasons')}` |"
    )
lines.extend(["", "## JSON", "", f"`{out_json}`", ""])
out_md.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"status": status, "out_md": str(out_md)}, indent=2))
PY
