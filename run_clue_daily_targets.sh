#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR"
LOG_FILE="${ROOT}/logs/daily_targets.log"
# 기본 배치 크기: 사용자 1~2명 단위로 먼저 운영하다가 점진 확장
BATCH_SIZE="${BATCH_SIZE:-1}"
BATCH_PAUSE_SEC="${BATCH_PAUSE_SEC:-30}"
MAX_RETRY="${MAX_RETRY:-1}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-0}"

# 기존 운영 대상(필요 시 배열만 확장)
TARGET_USERS=("주재욱")

mkdir -p "${ROOT}/logs"
cd "${ROOT}"

log_ts() {
  date '+%Y-%m-%d %H:%M:%S'
}

logf() {
  local msg="$1"
  echo "[$(log_ts)] ${msg}" >> "${LOG_FILE}"
}

resolve_code() {
  local name="$1"
  python3 - "$name" <<'PY'
import json
import sys
from pathlib import Path

name = sys.argv[1]
users = json.loads(Path('data/users/users.json').read_text(encoding='utf-8'))
for u in users:
    if u.get('name') == name and u.get('is_active', False) and u.get('user_code'):
        print(u['user_code'])
        break
PY
}

run_target() {
  local name="$1"
  local user_code="$2"
  local user_log="${ROOT}/logs/daily_targets.${user_code}.log"
  local attempt=0
  local rc=1

  while true; do
    attempt=$((attempt + 1))
    logf "RUN_START user=${name}(${user_code}) attempt=${attempt} log=${user_log}"

    if [[ "$RUN_TIMEOUT_SEC" -gt 0 ]]; then
      timeout_cmd=(timeout "$RUN_TIMEOUT_SEC")
    else
      timeout_cmd=()
    fi

    {
      ${timeout_cmd:+"${timeout_cmd[@]}"} \
      python3 service.py run --user-code "$user_code" --no-browser --trace
    } >> "$user_log" 2>&1
    rc=$?

    if [[ $rc -eq 0 ]]; then
      logf "RUN_SUCCESS user=${name}(${user_code}) attempt=${attempt}"
      return 0
    fi

    if [[ $attempt -gt $MAX_RETRY ]]; then
      logf "RUN_FAILED user=${name}(${user_code}) attempt=${attempt} rc=${rc}"
      return "$rc"
    fi

    logf "RETRY user=${name}(${user_code}) will retry attempt=$((attempt + 1))/${MAX_RETRY} after 10s"
    sleep 10
  done
}

run_batch() {
  local -n _users="$1"
  local batch_no="$2"
  local total="$3"
  local idx=0

  logf "BATCH_START batch=${batch_no}/${total} size=${#_users[@]}"
  for name in "${_users[@]}"; do
    idx=$((idx + 1))
    local user_code
    user_code="$(resolve_code "$name" || true)"
    user_code="$(echo "$user_code" | tr -d '\n')"

    if [[ -z "$user_code" ]]; then
      logf "BATCH_SKIP batch=${batch_no} idx=${idx} name=${name} reason=user_code_not_found_or_inactive"
      continue
    fi

    if ! run_target "$name" "$user_code"; then
      logf "BATCH_RUN_TARGET_ERROR batch=${batch_no} idx=${idx} name=${name} user_code=${user_code}"
    fi

    sleep 5
  done
  logf "BATCH_END batch=${batch_no}/${total}"
}

logf "cue-letter daily sender start"

if (( BATCH_SIZE < 1 )); then
  BATCH_SIZE=1
fi

CURRENT_BATCH=()
BATCH_NO=1
TOTAL_USERS=${#TARGET_USERS[@]}

for name in "${TARGET_USERS[@]}"; do
  CURRENT_BATCH+=("$name")
  if (( ${#CURRENT_BATCH[@]} >= BATCH_SIZE )); then
    run_batch CURRENT_BATCH "$BATCH_NO" "$(( (TOTAL_USERS + BATCH_SIZE - 1) / BATCH_SIZE ))"
    CURRENT_BATCH=()
    BATCH_NO=$((BATCH_NO + 1))
    if (( BATCH_NO <= ((TOTAL_USERS + BATCH_SIZE - 1) / BATCH_SIZE) )); then
      sleep "$BATCH_PAUSE_SEC"
    fi
  fi
 done

if (( ${#CURRENT_BATCH[@]} > 0 )); then
  run_batch CURRENT_BATCH "$BATCH_NO" "$(( (TOTAL_USERS + BATCH_SIZE - 1) / BATCH_SIZE ))"
fi

logf "cue-letter daily sender done"
