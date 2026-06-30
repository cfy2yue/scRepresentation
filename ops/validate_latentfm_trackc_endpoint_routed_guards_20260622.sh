#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

cd "${ROOT}"

bash -n \
  ops/launch_latentfm_trackc_endpoint_routed_uncapped_noharm_if_pass_20260622.sh \
  ops/summarize_latentfm_trackc_endpoint_routed_uncapped_noharm_20260622.sh \
  ops/launch_latentfm_trackc_routefocus_query_if_pass_20260622.sh \
  ops/launch_latentfm_trackc_endpoint_routed_query_if_pass_20260622.sh

"${PYTHON}" -m py_compile \
  ops/summarize_latentfm_trackc_routefocus_uncapped_noharm_20260622.py \
  ops/summarize_latentfm_trackc_routefocus_query_once_20260622.py

set +e
bash ops/launch_latentfm_trackc_endpoint_routed_uncapped_noharm_if_pass_20260622.sh >/tmp/endpoint_uncapped_guard.out 2>&1
uncapped_rc=$?
bash ops/launch_latentfm_trackc_endpoint_routed_query_if_pass_20260622.sh >/tmp/endpoint_query_guard.out 2>&1
query_rc=$?
set -e

if [[ "${uncapped_rc}" != "2" ]]; then
  cat /tmp/endpoint_uncapped_guard.out >&2
  echo "Expected endpoint uncapped guard RC=2 before smoke decision, got ${uncapped_rc}" >&2
  exit 10
fi
if [[ "${query_rc}" != "2" ]]; then
  cat /tmp/endpoint_query_guard.out >&2
  echo "Expected endpoint query guard RC=2 before smoke decision, got ${query_rc}" >&2
  exit 11
fi

unexpected=(
  reports/latentfm_trackc_endpoint_routed_uncapped_noharm_manifest_20260622.json
  reports/latentfm_trackc_endpoint_routed_uncapped_noharm_20260622
  runs/latentfm_trackc_endpoint_routed_uncapped_noharm_20260622
  reports/latentfm_trackc_endpoint_routed_uncapped_noharm_decision_20260622.json
  reports/LATENTFM_TRACKC_ENDPOINT_ROUTED_UNCAPPED_NOHARM_DECISION_20260622.md
  reports/latentfm_trackc_endpoint_routed_uncapped_noharm_bootstrap_20260622
  runs/latentfm_trackc_endpoint_routed_query_once_20260622
  reports/latentfm_trackc_endpoint_routed_query_once_20260622
  reports/latentfm_trackc_endpoint_routed_query_once_decision_20260622.json
  reports/LATENTFM_TRACKC_ENDPOINT_ROUTED_QUERY_ONCE_DECISION_20260622.md
  reports/latentfm_trackc_endpoint_routed_query_once_bootstrap_20260622
)
for path in "${unexpected[@]}"; do
  if [[ -e "${path}" ]]; then
    echo "Unexpected endpoint guard artifact exists: ${path}" >&2
    exit 12
  fi
done

rm -f /tmp/endpoint_uncapped_guard.out /tmp/endpoint_query_guard.out
echo "endpoint_routed_guard_validation_pass"
