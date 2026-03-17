#!/usr/bin/env bash
# Persistent-ready launcher for clue_letter
# Usage:
#   source /Users/davechoi/.openclaw/workspace/clue_letter/launch_clue.sh
# then run:
#   clue-run

# Load OPENAI_* env from repository .env files when present.
# Priority:
#  1) /Users/davechoi/.openclaw/workspace/openclaw/.env
#  2) ./clue_letter/.env
# Existing environment values are preserved.

set -a
if [ -f /Users/davechoi/.openclaw/workspace/openclaw/.env ]; then
  # shellcheck disable=SC1091
  source /Users/davechoi/.openclaw/workspace/openclaw/.env
fi
if [ -f /Users/davechoi/.openclaw/workspace/clue_letter/.env ]; then
  # shellcheck disable=SC1091
  source /Users/davechoi/.openclaw/workspace/clue_letter/.env
fi
set +a

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-mini}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"

clue-run() {
  cd /Users/davechoi/.openclaw/workspace/clue_letter
  python3 service.py run --user-code "$1" ${2:+"$2"}
}

echo "[clue_letter launcher loaded] Set OPENAI_MODEL=$OPENAI_MODEL"