from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append("/Users/davechoi/.openclaw/workspace")

from agents.super_agent import SuperAgent

ROOT = Path(__file__).resolve().parent


def _load_default_interests(name: str):
    presets = {
        "lcs": [
            "AI 인프라",
            "반도체",
            "데이터센터",
            "지정학 리스크",
            "GPU",
            "HBM",
            "클라우드",
            "AI 칩",
            "삼성",
            "SK hynix",
        ]
    }
    return presets.get(name.lower(), [])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["register", "interests", "run", "run-all", "list", "status"], help="action")
    parser.add_argument("--name", help="user name")
    parser.add_argument("--user-code", help="user code for run/run-status update")
    parser.add_argument("--interests", nargs="*", default=[], help="user interests")
    parser.add_argument("--countries", nargs="*", default=[], help="preferred countries")
    parser.add_argument("--exclusions", nargs="*", default=[], help="exclusions")
    parser.add_argument("--dry", action="store_true", help="dry-run only")
    parser.add_argument("--browser", action="store_true", help="use browser relay explicitly (default is HTTP/requests)")
    parser.add_argument("--no-browser", action="store_true", help="explicitly force no-browser; use HTTP/requests fetch path")

    args = parser.parse_args(argv)
    use_browser_relay = bool(args.browser and not args.no_browser)
    svc = SuperAgent(root=ROOT, use_browser_relay=use_browser_relay)

    if args.action == "register":
        if not args.name:
            raise SystemExit("--name required")
        interests = args.interests or _load_default_interests(args.name)
        user = svc.register_user(
            name=args.name,
            interests=interests,
            countries=args.countries,
            exclusions=args.exclusions,
        )
        print(json.dumps({"ok": True, "user_code": user.user_code, "name": user.name, "email": user.email}, ensure_ascii=False))
        return

    if args.action == "interests":
        if not args.user_code:
            raise SystemExit("--user-code required")
        updated = svc.update_interests(args.user_code, args.interests)
        print(json.dumps({"ok": True, "user_code": updated.user_code, "interests": updated.interests}, ensure_ascii=False))
        return

    if args.action == "list":
        rows = []
        for u in svc.list_users():
            rows.append({"name": u.name, "user_code": u.user_code, "interests": u.interests, "is_active": u.is_active})
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    if args.action == "run":
        if not args.user_code:
            raise SystemExit("--user-code required")
        out = svc.run_for_user(args.user_code, dry_run=args.dry)
        print(json.dumps(out, ensure_ascii=False))
        return

    if args.action == "run-all":
        out = svc.run_all(dry_run=args.dry)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.action == "status":
        if not args.user_code:
            raise SystemExit("--user-code required")
        u = svc.needs.get_user(args.user_code)
        print(json.dumps({"name": u.name, "user_code": u.user_code, "active": u.is_active, "interests": u.interests, "countries": u.countries}, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
