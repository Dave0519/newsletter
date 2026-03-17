from __future__ import annotations

import json
from pathlib import Path

from openclaw.agents.delivery import DeliveryManager

from .collection_agent import CollectionAgent
from .needs_agent import NeedsAgent
from .writing_agent import WritingAgent
from .browser_adapter import BrowserRelayAdapter


class SuperAgent:
    """실서비스 가능한 통합 오케스트레이터."""

    def __init__(
        self,
        root: Path,
        needs_file: Path | None = None,
        template_path: Path | None = None,
        use_browser_relay: bool = True,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.needs_file = needs_file or (self.root / "data" / "users" / "users.json")
        self.template_path = template_path or (self.root / "templates" / "CLUE_TEMPLATE_OFFICIAL.html")
        self.logger_root = self.root / "logs"
        self.logger_root.mkdir(parents=True, exist_ok=True)

        self.needs = NeedsAgent(self.needs_file)

        if use_browser_relay:
            adapter = BrowserRelayAdapter()
            core_urls = self._load_core_feed_urls()

            def _core_search(q: str, limit: int = 25):
                return adapter.search_core_rss(q, core_urls, limit=limit)

            self.collection = CollectionAgent(
                data_root=self.root / "data",
                needs_agent=self.needs,
                browser_search=adapter.search,
                fetch_body=adapter.fetch,
                resolve_url=adapter.resolve_google_news_url,
                search_core=_core_search,
                search_google=adapter.search_google_news,
                min_count=8,
            )
        else:
            self.collection = CollectionAgent(
                data_root=self.root / "data",
                needs_agent=self.needs,
                browser_search=None,
                fetch_body=None,
                min_count=8,
            )

        self.writer = WritingAgent(template_path=self.template_path, log_dir=self.logger_root)
        self.delivery = DeliveryManager()

    def register_user(
        self,
        name: str,
        interests: list[str],
        exclusions: list[str] | None = None,
        countries: list[str] | None = None,
        email: str = "bonggyu1.choi@sk.com",
        user_code: str | None = None,
    ):
        return self.needs.register_user(
            name=name,
            interests=interests,
            exclusions=exclusions,
            countries=countries,
            email=email,
            user_code=user_code,
        )

    def update_interests(self, user_code: str, interests: list[str]):
        return self.needs.set_user_interests(user_code=user_code, interests=interests)

    def list_users(self):
        return self.needs.list_users()

    def set_user_status(self, user_code: str, active: bool):
        self.needs.set_user_status(user_code=user_code, active=active)


    def _load_core_feed_urls(self) -> list[str]:
        cfg = self.root / "core_rss_active.json"
        if not cfg.exists():
            return []
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            return [item.get("url") for item in data.get("active", []) if item.get("url")]
        except Exception:
            return []
    def _next_issue_no(self, user_code: str) -> int:
        log = self.root / "logs" / "issue_no_tracker.json"
        if not log.exists():
            data = {}
        else:
            data = json.loads(log.read_text(encoding="utf-8"))
        current = int(data.get(user_code, 0)) + 1
        data[user_code] = current
        log.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return current

    def run_for_user(self, user_code: str, dry_run: bool = False, min_count: int = 8):
        user = self.needs.get_user(user_code)
        if not user.is_active:
            raise RuntimeError(f"user not active: {user_code}")

        articles = self.collection.collect(user=user, min_count=min_count)
        issue_no = self._next_issue_no(user.user_code)
        html_path = self.writer.build_and_save(user=user, collected=articles, issue_number=issue_no)

        if dry_run:
            return {
                "ok": True,
                "mode": "dry-run",
                "user_code": user.user_code,
                "issue_no": issue_no,
                "article_count": len(articles),
                "html_path": str(html_path),
            }

        delivery = self.delivery.deliver(html_path.read_text(encoding="utf-8"), f"{datetime_stamp()}", user.email)
        return {
            "ok": True,
            "mode": "mail",
            "user_code": user.user_code,
            "issue_no": issue_no,
            "article_count": len(articles),
            "html_path": str(html_path),
            "email": delivery,
        }

    def run_all(self, dry_run: bool = False, min_count: int = 8):
        out = []
        for user in self.needs.list_users():
            if not user.is_active:
                continue
            try:
                out.append(self.run_for_user(user.user_code, dry_run=dry_run, min_count=min_count))
            except Exception as e:
                out.append({
                    "ok": False,
                    "user_code": user.user_code,
                    "name": user.name,
                    "error": str(e),
                })
        return out


def datetime_stamp() -> str:
    return __import__("datetime").datetime.now().strftime("%Y-%m-%d")
