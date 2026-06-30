#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/cyx/1030/scLatent}"
HISTORY_FILE="${HISTORY_FILE:-${ROOT}/logs/progress_history.tsv}"
REPORT="${REPORT:-${ROOT}/reports/SYNC_PROGRESS.md}"
MIN_INTERVAL_SECONDS="${MIN_INTERVAL_SECONDS:-3600}"
EXPECTED_BIFLOW_GIB="${EXPECTED_BIFLOW_GIB:-137}"

to_gib() {
  local value="$1"
  awk -v v="$value" '
    BEGIN {
      if (v == "" || v == "NA") { print "nan"; exit }
      unit = substr(v, length(v), 1)
      num = substr(v, 1, length(v)-1) + 0
      if (unit == "K") print num / 1024 / 1024
      else if (unit == "M") print num / 1024
      else if (unit == "G") print num
      else if (unit == "T") print num * 1024
      else print v + 0
    }
  '
}

main() {
  mkdir -p "$(dirname "$REPORT")"
  if [[ ! -f "$HISTORY_FILE" ]]; then
    {
      printf '# Sync Progress\n\n'
      printf 'No progress history yet: `%s`\n' "$HISTORY_FILE"
    } >"$REPORT"
    echo "wrote ${REPORT}"
    return 0
  fi

  local rows
  rows="$(tail -n +2 "$HISTORY_FILE" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
  {
    printf '# Sync Progress\n\n'
    printf 'Generated: %s\n\n' "$(date '+%F %T')"
    printf 'Source history: `%s`\n\n' "$HISTORY_FILE"
    printf 'Recorded effective checks: `%s`\n\n' "$rows"
  } >"$REPORT"

  if (( rows == 0 )); then
    printf 'No effective progress checks have been recorded yet.\n' >>"$REPORT"
    echo "wrote ${REPORT}"
    return 0
  fi

  local first last
  first="$(tail -n +2 "$HISTORY_FILE" | sed '/^[[:space:]]*$/d' | head -n 1)"
  last="$(tail -n +2 "$HISTORY_FILE" | sed '/^[[:space:]]*$/d' | tail -n 1)"

  IFS=$'\t' read -r first_ts _ _ _ first_dataset first_biflow _ _ _ _ _ _ <<<"$first"
  IFS=$'\t' read -r last_ts last_transfer last_watcher last_status last_dataset last_biflow last_pretrained last_third last_pretrainckpt last_mem last_free last_load <<<"$last"

  local first_gib last_gib delta_gib first_epoch last_epoch hours rate
  first_gib="$(to_gib "$first_dataset")"
  last_gib="$(to_gib "$last_dataset")"
  first_epoch="$(date -d "$first_ts" +%s)"
  last_epoch="$(date -d "$last_ts" +%s)"
  hours="$(awk -v a="$first_epoch" -v b="$last_epoch" 'BEGIN { printf "%.2f", (b-a)/3600 }')"
  delta_gib="$(awk -v a="$first_gib" -v b="$last_gib" 'BEGIN { printf "%.1f", b-a }')"
  rate="$(awk -v d="$delta_gib" -v h="$hours" 'BEGIN { if (h > 0) printf "%.1f", d/h; else printf "NA" }')"
  local next_epoch next_check remaining_biflow eta_hours
  next_epoch="$((last_epoch + MIN_INTERVAL_SECONDS))"
  next_check="$(date -d "@$next_epoch" '+%F %T')"
  remaining_biflow="$(awk -v expected="$EXPECTED_BIFLOW_GIB" -v current="$(to_gib "$last_biflow")" 'BEGIN { r=expected-current; if (r < 0) r=0; printf "%.1f", r }')"
  eta_hours="$(awk -v rem="$remaining_biflow" -v r="$rate" 'BEGIN { if (r > 0) printf "%.1f", rem/r; else printf "NA" }')"

  {
    printf '## Latest Effective Check\n\n'
    printf '| Field | Value |\n'
    printf '| --- | --- |\n'
    printf '| Timestamp | `%s` |\n' "$last_ts"
    printf '| Transfer alive | `%s` |\n' "$last_transfer"
    printf '| Watcher alive | `%s` |\n' "$last_watcher"
    printf '| Status | `%s` |\n' "$last_status"
    printf '| Dataset size | `%s` |\n' "$last_dataset"
    printf '| biFlow size | `%s` |\n' "$last_biflow"
    printf '| scFM pretrained | `%s` |\n' "$last_pretrained"
    printf '| scFM third party | `%s` |\n' "$last_third"
    printf '| pretrainckpt | `%s` |\n' "$last_pretrainckpt"
    printf '| MemAvailable GiB | `%s` |\n' "$last_mem"
    printf '| Data free | `%s` |\n' "$last_free"
    printf '| Load average | `%s` |\n' "$last_load"
    printf '| Next allowed check | `%s` |\n\n' "$next_check"

    printf '## Observed Growth\n\n'
    printf '| Window | Dataset growth | Approx rate |\n'
    printf '| --- | ---: | ---: |\n'
    printf '| `%s` to `%s` (`%sh`) | `%s GiB` | `%s GiB/h` |\n\n' \
      "$first_ts" "$last_ts" "$hours" "$delta_gib" "$rate"

    printf '## Rough ETA\n\n'
    printf '| Resource | Expected size | Current size | Remaining | ETA at observed rate |\n'
    printf '| --- | ---: | ---: | ---: | ---: |\n'
    printf '| CoupledFM biFlow stack data | `%s GiB` | `%s` | `%s GiB` | `%s h` |\n\n' \
      "$EXPECTED_BIFLOW_GIB" "$last_biflow" "$remaining_biflow" "$eta_hours"

    printf '## Recent Rows\n\n'
    printf '```text\n'
    tail -n 6 "$HISTORY_FILE"
    printf '```\n'
  } >>"$REPORT"

  echo "wrote ${REPORT}"
}

main "$@"
