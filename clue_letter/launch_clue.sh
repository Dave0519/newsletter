#!/usr/bin/env bash
# Persistent-ready launcher for clue_letter
# Usage:
#   source /Users/davechoi/.openclaw/workspace/clue_letter/launch_clue.sh
# then run:
#   clue-run

export OPENAI_API_KEY="${OPENAI_API_KEY:-}" # set this externally
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-mini}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"

clue-run() {
  cd /Users/davechoi/.openclaw/workspace/clue_letter
  python3 service.py run --user-code "$1" ${2:+"$2"}
}

echo "[clue_letter launcher loaded] Set OPENAI_MODEL=$OPENAI_MODEL"
