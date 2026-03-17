from __future__ import annotations

import sys
sys.path.append("/Users/davechoi/.openclaw/workspace")

from pathlib import Path

from agents.super_agent import SuperAgent

from agents.super_agent import SuperAgent

ROOT = Path('/Users/davechoi/.openclaw/workspace/clue_letter')


def main():
    svc = SuperAgent(root=ROOT, use_browser_relay=True)
    # default user registration (idempotent)
    user = svc.register_user(
        name="lcs",
        interests=[
            "AI 인프라",
            "반도체",
            "데이터센터",
            "지정학 리스크",
            "GPU",
            "HBM",
            "클라우드",
            "중동",
            "수급",
            "AI 칩",
            "삼성",
            "SK hynix",
        ],
        countries=["KR", "US", "CN", "TW", "GLOBAL"],
    )

    print("registered_or_existing:", user.user_code)

    # production mode dry-run by default in test run
    result = svc.run_for_user(user.user_code, dry_run=False)
    print(result)


if __name__ == '__main__':
    main()