#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

ACK=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_ACK:-}
if [[ "${ACK}" != "pairtype_support_gate_pass_external_review" && "${ACK}" != "target_population_external_review_noharm_veto_only" ]]; then
  cat >&2 <<'EOF'
Refusing to launch Track C support-only uncapped no-harm.

Set:
  LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_ACK=pairtype_support_gate_pass_external_review
or, for the narrowed target-population route only:
  LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_ACK=target_population_external_review_noharm_veto_only

Boundary:
  - requires a query-free support-only robustness decision pass
  - target-population route also requires explicit external review and seed45 whole-support fail acknowledgement
  - canonical split_seed42 single/family no-harm only
  - canonical multi may be produced only as diagnostic output; selection weight = 0
  - held-out Track C query remains forbidden
  - support context is forced absent for canonical no-harm
EOF
  exit 4
fi

RUN_NAME=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_RUN_NAME:-}
if [[ -z "${RUN_NAME}" ]]; then
  echo "Set LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_RUN_NAME to the passed support-only run name." >&2
  exit 2
fi

RUN_ROOT=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_SOURCE_RUN_ROOT:-${ROOT}/runs/latentfm_trackc_support_only_robustness_20260624/${RUN_NAME}}
OUT_ROOT=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_SOURCE_OUT_ROOT:-${COUPLED}/output/latentfm_runs/trackc_support_only_robustness_20260624}
DECISION_JSON=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_SOURCE_DECISION_JSON:-${ROOT}/reports/latentfm_trackc_support_only_robustness_decision_${RUN_NAME}.json}
STRATA_SUMMARY_JSON=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_STRATA_SUMMARY_JSON:-${ROOT}/reports/latentfm_trackc_support_only_pairtype_strata_summary_20260624.json}
EXPECTED_PAIR_TYPE_FILTER=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_PAIR_TYPE_FILTER:-none_train_single_both_train_multi_gene}
EXPECTED_STABILITY_STATUS=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_STABILITY_STATUS:-pass_2_of_3_no_hard_fail}
TARGETPOP_GATE_JSON=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_TARGETPOP_GATE_JSON:-}
TARGETPOP_EXTERNAL_REVIEW_MD=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_TARGETPOP_EXTERNAL_REVIEW_MD:-}
TARGETPOP_ADJUDICATION_JSON=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_TARGETPOP_ADJUDICATION_JSON:-}
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
CANDIDATE_CKPT=${OUT_ROOT}/${RUN_NAME}/best.pt
CONFIG_JSON=${OUT_ROOT}/${RUN_NAME}/config.json
CANONICAL_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42.json
SAFE_TRAINSELECT_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
SAFE_RUN_ID=$(printf '%s' "${RUN_NAME}" | tr -c 'A-Za-z0-9_' '_')
LABEL=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_LABEL:-latentfm_trackc_support_only_uncapped_noharm_${SAFE_RUN_ID}_20260624}
MANIFEST=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_MANIFEST:-${ROOT}/reports/${LABEL}_manifest.json}
UNCAPPED_OUT=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_OUT_DIR:-${ROOT}/reports/${LABEL}}
DECISION_OUT_JSON=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_DECISION_JSON:-${ROOT}/reports/${LABEL}_decision.json}
DECISION_OUT_MD=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_DECISION_MD:-${ROOT}/reports/LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_NOHARM_${SAFE_RUN_ID}_DECISION_20260624.md}
BOOT_DIR=${LATENTFM_TRACKC_SUPPORT_ONLY_UNCAPPED_BOOT_DIR:-${ROOT}/reports/${LABEL}_bootstrap}
LAUNCHER=${ROOT}/ops/launch_latentfm_uncapped_posthoc_from_manifest_20260621.sh
SUMMARY_WRAPPER=${ROOT}/ops/summarize_latentfm_trackc_support_context_v2_uncapped_noharm_20260623.sh

for required in \
  "${DECISION_JSON}" \
  "${STRATA_SUMMARY_JSON}" \
  "${RUN_ROOT}/RUN_STATUS.md" \
  "${ANCHOR_CKPT}" \
  "${CANDIDATE_CKPT}" \
  "${CONFIG_JSON}" \
  "${ROOT}/dataset/latentfm_full/xverse/manifest.json" \
  "${CANONICAL_SPLIT}" \
  "${LAUNCHER}" \
  "${SUMMARY_WRAPPER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

if [[ "${ACK}" == "target_population_external_review_noharm_veto_only" ]]; then
  for required in \
    "${TARGETPOP_GATE_JSON}" \
    "${TARGETPOP_EXTERNAL_REVIEW_MD}" \
    "${TARGETPOP_ADJUDICATION_JSON}"; do
    if [[ ! -e "${required}" ]]; then
      echo "Missing target-population route artifact: ${required}" >&2
      exit 2
    fi
  done
fi

"${PYTHON}" - "${DECISION_JSON}" "${STRATA_SUMMARY_JSON}" "${CONFIG_JSON}" "${MANIFEST}" "${ANCHOR_CKPT}" "${CANDIDATE_CKPT}" "${RUN_ROOT}" "${RUN_NAME}" "${CANONICAL_SPLIT}" "${SAFE_TRAINSELECT_SPLIT}" "${EXPECTED_PAIR_TYPE_FILTER}" "${EXPECTED_STABILITY_STATUS}" "${ACK}" "${TARGETPOP_GATE_JSON}" "${TARGETPOP_EXTERNAL_REVIEW_MD}" "${TARGETPOP_ADJUDICATION_JSON}" <<'PY'
import json
import re
import sys
from pathlib import Path

decision_json = Path(sys.argv[1])
strata_summary_json = Path(sys.argv[2])
config_json = Path(sys.argv[3])
manifest = Path(sys.argv[4])
anchor_ckpt = Path(sys.argv[5])
candidate_ckpt = Path(sys.argv[6])
run_root = Path(sys.argv[7])
run_name = sys.argv[8]
canonical_split = Path(sys.argv[9])
safe_trainselect_split = Path(sys.argv[10])
expected_pair_type_filter = sys.argv[11]
expected_stability_status = sys.argv[12]
ack = sys.argv[13]
targetpop_gate_json = Path(sys.argv[14]) if sys.argv[14] else None
targetpop_external_review_md = Path(sys.argv[15]) if sys.argv[15] else None
targetpop_adjudication_json = Path(sys.argv[16]) if sys.argv[16] else None

def seed_number(name: str) -> int:
    m = re.search(r"_seed(\d+)(?:$|_)", name)
    if not m:
        return 10**9
    return int(m.group(1))

decision = json.loads(decision_json.read_text(encoding="utf-8"))
status = (decision.get("decision") or {}).get("status") or decision.get("status")
if status != "trackc_support_only_robustness_pass_support_gate":
    raise SystemExit(
        json.dumps(
            {
                "status": "support_only_decision_not_pass",
                "decision_status": status,
                "decision_json": str(decision_json),
            },
            indent=2,
        )
    )
boundary = decision.get("boundary") or {}
if boundary.get("heldout_query_read") is not False:
    raise SystemExit("support decision boundary is not query-blind")
if boundary.get("canonical_metrics_read") is not False:
    raise SystemExit("support decision unexpectedly read canonical metrics")
if boundary.get("canonical_multi_selection") is not False:
    raise SystemExit("support decision unexpectedly used canonical multi selection")
if Path(str(boundary.get("expected_split_file", ""))) != safe_trainselect_split:
    raise SystemExit(f"support decision split mismatch: {boundary.get('expected_split_file')}")

strata = json.loads(strata_summary_json.read_text(encoding="utf-8"))
stability_status = strata.get("stability_status") or strata.get("status")
if stability_status != expected_stability_status:
    raise SystemExit(
        json.dumps(
            {
                "status": "seed_stability_not_passed",
                "stability_status": stability_status,
                "expected_stability_status": expected_stability_status,
                "strata_summary_json": str(strata_summary_json),
            },
            indent=2,
        )
    )
passing = [
    str(row.get("run_name"))
    for row in strata.get("runs", [])
    if row.get("status") == "pass_pairtype_target_support_control"
]
if not passing:
    raise SystemExit("seed stability summary has no passing runs")
frozen = sorted(passing, key=seed_number)[0]
if run_name != frozen:
    raise SystemExit(
        json.dumps(
            {
                "status": "run_not_frozen_lowest_passing_seed",
                "requested_run": run_name,
                "frozen_run": frozen,
                "passing_runs": passing,
            },
            indent=2,
        )
    )

config = json.loads(config_json.read_text(encoding="utf-8"))
if config.get("split_file") != str(safe_trainselect_split):
    raise SystemExit(f"training config split_file mismatch: {config.get('split_file')}")
if config.get("trackc_support_context_pair_type_filter") != expected_pair_type_filter:
    raise SystemExit(
        "uncapped no-harm wrapper expected a different predeclared pair-type filter; "
        f"expected {expected_pair_type_filter}, got {config.get('trackc_support_context_pair_type_filter')}"
    )
if config.get("trackc_support_film_use_in_model") is not True:
    raise SystemExit("candidate config does not enable support FiLM")
if config.get("finetune_trainable_scope") != "support_film_adapter":
    raise SystemExit(f"unexpected trainable scope: {config.get('finetune_trainable_scope')}")
if manifest.exists():
    raise FileExistsError(f"refusing to overwrite existing manifest: {manifest}")

targetpop_route = None
if ack == "target_population_external_review_noharm_veto_only":
    if expected_pair_type_filter != "both_train_multi_gene":
        raise SystemExit(
            "target-population route is only authorized for pair-type filter both_train_multi_gene"
        )
    target_gate = json.loads(targetpop_gate_json.read_text(encoding="utf-8"))
    if target_gate.get("status") != "trackc_target_population_support_gate_pass_external_review_next":
        raise SystemExit(
            json.dumps(
                {
                    "status": "target_population_gate_not_pass",
                    "target_gate_status": target_gate.get("status"),
                    "targetpop_gate_json": str(targetpop_gate_json),
                },
                indent=2,
            )
        )
    if target_gate.get("gpu_authorized") is not False:
        raise SystemExit("target-population gate should authorize external review only, not GPU directly")
    if target_gate.get("whole_support_adjudication_status") != "trackc_both_train_multi_gene_adjudication_fail_no_gpu":
        raise SystemExit("target-population gate did not preserve whole-support adjudication failure")
    target_rows = [r for r in target_gate.get("rows", []) if r.get("pass") is True]
    if len(target_rows) != len(target_gate.get("rows", [])) or len(target_rows) < 3:
        raise SystemExit("target-population route requires all three seed rows to pass")
    target_passing = [str(r.get("run")) for r in target_rows]
    frozen_target = sorted(target_passing, key=seed_number)[0]
    if run_name != frozen_target:
        raise SystemExit(
            json.dumps(
                {
                    "status": "run_not_frozen_lowest_target_population_seed",
                    "requested_run": run_name,
                    "frozen_target_seed": frozen_target,
                    "target_passing_runs": target_passing,
                },
                indent=2,
            )
        )

    review_text = targetpop_external_review_md.read_text(encoding="utf-8")
    if "Status: `pass_conditional_noharm_veto_only`" not in review_text:
        raise SystemExit("external review did not record pass_conditional_noharm_veto_only")
    if "Held-out Track C query remains forbidden" not in review_text:
        raise SystemExit("external review did not preserve held-out query boundary")

    adjudication = json.loads(targetpop_adjudication_json.read_text(encoding="utf-8"))
    if adjudication.get("status") != "trackc_both_train_multi_gene_adjudication_fail_no_gpu":
        raise SystemExit("adjudication gate status was not the expected fail/no-gpu")
    seed45 = [r for r in adjudication.get("rows", []) if int(r.get("seed", -1)) == 45]
    if len(seed45) != 1:
        raise SystemExit("adjudication gate did not contain exactly one seed45 row")
    seed45_row = seed45[0]
    if seed45_row.get("whole_pass") is not False or seed45_row.get("whole_hard_or_materiality_fail") is not True:
        raise SystemExit("seed45 whole-support fail was not preserved in adjudication gate")
    if float(seed45_row.get("actual_pp", 1.0)) >= 0.04:
        raise SystemExit("seed45 actual_pp no longer records materiality fail below +0.04")
    targetpop_route = {
        "scope": "frozen canonical test_single/family_gene no-harm veto only",
        "target_population_gate_json": str(targetpop_gate_json),
        "target_population_gate_status": target_gate.get("status"),
        "external_review_report": str(targetpop_external_review_md),
        "external_review_status": "pass_conditional_noharm_veto_only",
        "adjudication_json": str(targetpop_adjudication_json),
        "whole_support_adjudication_status": adjudication.get("status"),
        "whole_support_fail_acknowledged": True,
        "seed45_whole_support_materiality_fail": {
            "actual_pp": seed45_row.get("actual_pp"),
            "threshold": 0.04,
        },
        "not_authorized": [
            "held_out_trackc_query",
            "canonical_multi_selection_or_success_claim",
            "whole_support_rescue_claim",
            "promotion_claim",
        ],
    }

out = {
    "purpose": "Track C support-only pair-type uncapped canonical no-harm; no held-out query",
    "split_file": str(canonical_split),
    "data_dir": "/data/cyx/1030/dataset/latentfm_full/xverse",
    "biflow_dir": "/data/cyx/1030/dataset/biFlow_data",
    "anchor_checkpoint": str(anchor_ckpt),
    "force_support_context_absent": True,
    "heldout_query_used": False,
    "authorized_scope": (
        "frozen canonical test_single/family_gene no-harm veto only"
        if targetpop_route
        else "support-only pair-type canonical no-harm"
    ),
    "canonical_multi_selection_weight": 0,
    "canonical_multi_diagnostic_only": True,
    "target_population_route": targetpop_route,
    "source_support_decision_json": str(decision_json),
    "source_strata_summary_json": str(strata_summary_json),
    "seed_stability_status": stability_status,
    "expected_pair_type_filter": expected_pair_type_filter,
    "frozen_seed_rule": strata.get("frozen_seed_rule"),
    "source_run_root": str(run_root),
    "launched_runs": [
        {
            "run_name": run_name,
            "candidate_checkpoint": str(candidate_ckpt),
            "split_file": str(canonical_split),
            "data_dir": "/data/cyx/1030/dataset/latentfm_full/xverse",
            "biflow_dir": "/data/cyx/1030/dataset/biFlow_data",
            "force_support_context_absent": True,
            "source_support_decision_json": str(decision_json),
            "source_strata_summary_json": str(strata_summary_json),
            "source_training_config_json": str(config_json),
            "source_support_pair_type_filter": config.get("trackc_support_context_pair_type_filter"),
        }
    ],
}
manifest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": "manifest_written", "manifest": str(manifest)}, indent=2))
PY

MANIFEST="${MANIFEST}" \
LABEL="${LABEL}" \
OUT_DIR="${UNCAPPED_OUT}" \
ONLY_RUN_NAME="${RUN_NAME}" \
EVAL_MAX_MSE_CELLS="${EVAL_MAX_MSE_CELLS:-0}" \
EVAL_MAX_MMD_CELLS="${EVAL_MAX_MMD_CELLS:-0}" \
SPLIT_GROUPS="test_single" \
FAMILY_GROUPS="family_gene" \
bash "${LAUNCHER}"

cat <<EOF

After uncapped posthoc finishes, summarize no-harm with:
LATENTFM_TRACKC_V2_UNCAPPED_LABEL='${LABEL}' \\
LATENTFM_TRACKC_V2_UNCAPPED_INDEX_JSON='${UNCAPPED_OUT}/uncapped_posthoc_index.json' \\
LATENTFM_TRACKC_V2_UNCAPPED_OUT_JSON='${DECISION_OUT_JSON}' \\
LATENTFM_TRACKC_V2_UNCAPPED_OUT_MD='${DECISION_OUT_MD}' \\
LATENTFM_TRACKC_V2_UNCAPPED_BOOT_DIR='${BOOT_DIR}' \\
LATENTFM_TRACKC_V2_UNCAPPED_REPORT_TITLE='LatentFM Track C Support-Only Pair-Type Uncapped Canonical No-Harm Decision: ${RUN_NAME}' \\
LATENTFM_TRACKC_V2_UNCAPPED_SPLIT_GROUPS='test_single' \\
LATENTFM_TRACKC_V2_UNCAPPED_FAMILY_GROUPS='family_gene' \\
bash ${SUMMARY_WRAPPER}
EOF
