#!/usr/bin/env bash
# Persistent-ready launcher for clue_letter
# Usage:
#   source <clue_letter_root>/launch_clue.sh
# then run:
#   clue-run

# Load OPENAI_* env from repository .env files when present.
# Priority:
#  1) <workspace>/openclaw/.env
#  2) <clue_letter_dev2>/.env
# Existing environment values are preserved.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

set -a
if [ -f "$WORKSPACE_ROOT/openclaw/.env" ]; then
  # shellcheck disable=SC1091
  source "$WORKSPACE_ROOT/openclaw/.env"
fi
if [ -f "$SCRIPT_DIR/.env" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
fi
set +a

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-mini}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"

clue-run() {
  cd "$SCRIPT_DIR"
  python3 service.py run --user-code "$1" ${2:+"$2"}
}

echo "[clue_letter launcher loaded] Set OPENAI_MODEL=$OPENAI_MODEL"