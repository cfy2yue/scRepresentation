#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/cyx/1030/scLatent}"
REMOTE="${REMOTE:-LiLab}"
REMOTE_ROOT="${REMOTE_ROOT:-/data2/cfy/FM/CoupledFM}"
REPORT="${REPORT:-${ROOT}/reports/FINAL_RSYNC_AUDIT.md}"
TRANSFER_STATUS_FILE="${TRANSFER_STATUS_FILE:-${ROOT}/logs/transfer_from_lilab.status}"
TRANSFER_LOG="${TRANSFER_LOG:-${ROOT}/logs/transfer_from_lilab.log}"
MAX_REPORT_LINES="${MAX_REPORT_LINES:-200}"

require_sync_complete() {
  if ! grep -q $'\tALL DONE' "$TRANSFER_STATUS_FILE" 2>/dev/null && \
     ! grep -q 'ALL DONE' "$TRANSFER_LOG" 2>/dev/null; then
    echo "transfer has not reported ALL DONE; final rsync audit is intentionally disabled" >&2
    exit 10
  fi
}

run_low_priority() {
  if command -v ionice >/dev/null 2>&1; then
    nice -n 5 ionice -c 2 -n 4 "$@"
  else
    nice -n 5 "$@"
  fi
}

audit_one() {
  local label="$1"
  local src="$2"
  local dst="$3"
  shift 3
  local tmp
  tmp="$(mktemp)"
  local rc=0
  run_low_priority rsync -ani --delete "$@" "${REMOTE}:${src%/}/" "${dst%/}/" >"$tmp" 2>&1 || rc=$?
  local changes
  changes="$(wc -l <"$tmp" | tr -d ' ')"
  {
    printf '### %s\n\n' "$label"
    printf -- '- Source: `%s:%s`\n' "$REMOTE" "$src"
    printf -- '- Destination: `%s`\n' "$dst"
    printf -- '- rsync exit code: `%s`\n' "$rc"
    printf -- '- Itemized differences: `%s`\n\n' "$changes"
    printf '```text\n'
    head -n "$MAX_REPORT_LINES" "$tmp"
    if (( changes > MAX_REPORT_LINES )); then
      printf '\n... truncated after %s lines ...\n' "$MAX_REPORT_LINES"
    fi
    printf '```\n\n'
  } >>"$REPORT"
  rm -f "$tmp"
  return "$rc"
}

main() {
  require_sync_complete
  mkdir -p "$(dirname "$REPORT")"
  {
    printf '# Final Rsync Audit\n\n'
    printf 'Generated: %s\n\n' "$(date '+%F %T')"
    printf 'This report uses `rsync -ani --delete` to compare LiLab sources with local copies after transfer completion.\n\n'
    printf 'Interpretation notes:\n\n'
    printf -- '- Directory-only timestamp differences such as `.d..t......` are not content mismatches.\n'
    printf -- '- `scFM_output/embedding_runs/{manifest.jsonl,manifest_with_X.jsonl,preflight.json}` may differ because local validation regenerates them with `/data/cyx/1030/scLatent` paths.\n'
    printf -- '- Third-party Python bytecode caches are excluded from the comparison.\n\n'
  } >"$REPORT"

  local failed=0
  audit_one "CoupledFM biFlow stack data" \
    "${REMOTE_ROOT}/model/local_biFlow_data/biFlow_data" \
    "${ROOT}/dataset/biFlow_data" || failed=1
  audit_one "CoupledFM cellgene census processed data" \
    "${REMOTE_ROOT}/model/local_biFlow_data/cellgene_census" \
    "${ROOT}/dataset/cellgene_census" || failed=1
  audit_one "scFMBench staging data" \
    "${REMOTE_ROOT}/scFM/data" \
    "${ROOT}/dataset/scFM_data" \
    --exclude='*.tmp.h5ad' \
    --exclude='*.bak*' \
    --exclude='*.before_*' || failed=1
  audit_one "raw source h5ad data" \
    "${REMOTE_ROOT}/data/raw" \
    "${ROOT}/dataset/raw" || failed=1
  audit_one "scFM pretrained assets" \
    "${REMOTE_ROOT}/scFM/pretrained" \
    "${ROOT}/scFM_pretrained" || failed=1
  audit_one "CoupledFM CellNavi gene cache" \
    "${REMOTE_ROOT}/model/condition_emb/genepert/cache/cellnavi_embed_gene" \
    "${ROOT}/pretrainckpt/genepert_cache/cellnavi_embed_gene" || failed=1
  audit_one "CoupledFM scGPT gene cache" \
    "${REMOTE_ROOT}/model/condition_emb/genepert/cache/scgpt_embed_gene" \
    "${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene" || failed=1
  audit_one "scFM third-party sources" \
    "${REMOTE_ROOT}/scFM/fm/third_party" \
    "${ROOT}/scFM_third_party" \
    --exclude='__pycache__/' \
    --exclude='*.pyc' || failed=1
  audit_one "scFM output run metadata" \
    "${REMOTE_ROOT}/scFM/output/embedding_runs" \
    "${ROOT}/scFM_output/embedding_runs" || failed=1
  audit_one "scFM benchmark inventory" \
    "${REMOTE_ROOT}/scFM/output/benchmark_inventory" \
    "${ROOT}/scFM_output/benchmark_inventory" || failed=1

  if [[ "$failed" == "0" ]]; then
    printf 'Final rsync audit completed. See %s\n' "$REPORT"
  else
    printf 'Final rsync audit completed with rsync errors. See %s\n' "$REPORT" >&2
    return 20
  fi
}

main "$@"
