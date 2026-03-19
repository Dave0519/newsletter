from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        return


def _bootstrap_env() -> None:
    # Runtime safety: ensure OpenAI keys are loaded even without shell source.
    root = Path(__file__).resolve().parent
    _load_env_file((root.parent) / "openclaw" / ".env")
    _load_env_file(root / ".env")


_bootstrap_env()

os.environ.setdefault("OPENAI_MODEL", "gpt-5-mini")
os.environ.setdefault("OPENAI_API_BASE", "https://api.openai.com/v1")

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
    parser.add_argument("--trace", action="store_true", help="enable detailed API/step trace logs")
    parser.add_argument("--batch-size", type=int, default=0, help="Run only first N users from selected batch")
    parser.add_argument("--batch-start", type=int, default=0, help="Zero-based start index for user batch")
    parser.add_argument("--browser", action="store_true", help="use browser relay explicitly (default is HTTP/requests)")
    parser.add_argument("--no-browser", action="store_true", help="explicitly force no-browser; use HTTP/requests fetch path")

    args = parser.parse_args(argv)

    if args.trace:
        os.environ["CLUE_TRACE"] = "1"
    trace_dir = Path("logs") / "trace"
    os.environ.setdefault("CLUE_TRACE_DIR", str(ROOT / trace_dir))
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
        out = _run_all_batched(svc, dry_run=args.dry, batch_size=args.batch_size, batch_start=args.batch_start)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.action == "status":
        if not args.user_code:
            raise SystemExit("--user-code required")
        u = svc.needs.get_user(args.user_code)
        print(json.dumps({"name": u.name, "user_code": u.user_code, "active": u.is_active, "interests": u.interests, "countries": u.countries}, ensure_ascii=False, indent=2))
        return




def _run_all_batched(svc: SuperAgent, dry_run: bool, batch_size: int, batch_start: int, min_count: int = 8):
    users = [u for u in svc.list_users() if u.is_active]
    if not users:
        return []
    start = max(0, batch_start)
    end = len(users) if batch_size <= 0 else min(len(users), start + batch_size)
    codes = [u.user_code for u in users[start:end]]

    out = []
    for code in codes:
        try:
            out.append(svc.run_for_user(code, dry_run=dry_run, min_count=min_count))
        except Exception as e:
            out.append({"ok": False, "user_code": code, "name": next((u.name for u in users if u.user_code == code), None), "error": str(e)})
    return out

if __name__ == "__main__":
    main()
