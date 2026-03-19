#!/usr/bin/env bash
set -euo pipefail

# clue_letter_dev2 정책/실행 구조 변경분을 빠르게 커밋+푸시하는 도우미
# 사용법: ./auto_sync_clue.sh "[optional message]"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

NOW="$(date '+%Y-%m-%d %H:%M:%S')"
MSG="${1:-chore(clue_letter_dev2): sync policy and runtime changes ${NOW}}"

# clue_letter_dev2 변경 감지 (운영 코드 기준)
CHANGED=$(git status --short | awk '{print $2}' | grep -E '^agents/|^core_rss_|^service.py$|^launch_clue.sh$|^run_test.py$|^run_test\.py$|^templates/|^README.md$|^standard_policy.json$|^daily_news.schema.json$|^total_news.schema.json$' || true)

if [[ -z "$CHANGED" ]]; then
  echo "No clue_letter_dev2 policy/runtime changes to sync."
  exit 0
fi

echo "Changes detected for clue_letter_dev2:" 
echo "$CHANGED"

# 운영 코드만 커밋 대상
for f in $CHANGED; do
  case "$f" in
    data/*|logs/*) ;;
    *) git add "$f" ;;
  esac
done

if git diff --cached --quiet; then
  echo "Nothing staged after filtering (only runtime data outputs)."
  exit 0
fi

git commit -m "$MSG"

git push origin main

COMMIT_HASH=$(git rev-parse --short HEAD)
echo "Synced: $COMMIT_HASH"
