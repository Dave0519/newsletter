from __future__ import annotations

import sys
import argparse
from pathlib import Path


def _discover_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(6):
        if (cur / "README.md").exists() and (cur / "agents").exists() and (cur / "service.py").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start


REPO_ROOT = _discover_repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

ROOT = Path(__file__).resolve().parent

from agents.super_agent import SuperAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", help="actually deliver (default: dry-run)")
    args = parser.parse_args()

    svc = SuperAgent(root=ROOT, use_browser_relay=False)
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

    # production mode is dry-run by default in test run
    result = svc.run_for_user(user.user_code, dry_run=not args.send)
    print(result)


if __name__ == "__main__":
    main()