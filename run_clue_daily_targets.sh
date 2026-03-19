#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/davechoi/.openclaw/workspace/clue_letter_dev2"
LOG_FILE="${ROOT}/logs/daily_targets.log"
TARGET_USERS=("주재욱" "송창석")
TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

mkdir -p "${ROOT}/logs"
cd "${ROOT}"

echo "[${TIMESTAMP}] cue-letter daily sender start" >> "${LOG_FILE}"

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
  local user_code
  user_code="$(resolve_code "$name")"

  if [[ -z "$user_code" ]]; then
    echo "[${TIMESTAMP}] SKIP: ${name} not found or inactive in data/users/users.json" >> "${LOG_FILE}"
    return 0
  fi

  echo "[${TIMESTAMP}] RUN: ${name} (${user_code})" >> "${LOG_FILE}"
  python3 service.py run --user-code "$user_code" --no-browser >> "${LOG_FILE}" 2>&1
}

for name in "${TARGET_USERS[@]}"; do
  run_target "$name"
done

echo "[${TIMESTAMP}] cue-letter daily sender done" >> "${LOG_FILE}"
