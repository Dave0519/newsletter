from __future__ import annotations

import json
import time
import threading
import os
from datetime import datetime
from pathlib import Path

try:
    from openclaw.agents.delivery import DeliveryManager as _OpenclawDeliveryManager
    _HAS_OPENCLAW_DELIVERY = True
except Exception:
    _OpenclawDeliveryManager = None
    _HAS_OPENCLAW_DELIVERY = False

from .collection_agent import CollectionAgent
from .needs_agent import NeedsAgent
from .writing_agent import WritingAgent
from .browser_adapter import BrowserRelayAdapter, HttpNewsAdapter




def _delivery_mode() -> str:
    mode = (
        os.getenv("SKNOW_DELIVERY_MODE", "openclaw")
        .strip()
        .lower()
        .replace("_", "-")
    )
    if mode in {"fallback", "local", "local-fallback", "local_fallback"}:
        return "fallback"
    if mode in {"openclaw-only", "openclaw-only-strict", "strict", "strict-gog", "openclaw-strict"}:
        return "openclaw-strict"
    # gog/openclaw should be compatible mode: try openclaw delivery when available,
    # otherwise degrade to local file fallback instead of hard-failing.
    if mode in {"gog", "openclaw", "openclaw-compatible"}:
        return "openclaw"
    return "openclaw"


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, "") or default))
    except Exception:
        return max(1, default)


class LocalFileDelivery:
    """Fallback delivery when openclaw.delivery is not available."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def deliver(self, html: str, subject: str, to: str) -> dict:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.root / f"delivery_{ts}.html"
        out.write_text(html, encoding="utf-8")
        return {
            "ok": True,
            "mode": "local-fallback",
            "subject": subject,
            "to": to,
            "out": str(out),
            "warning": "openclaw delivery module is unavailable. saved html only.",
        }


class NullDeliveryManager:
    """Explicit fallback that blocks outbound send but keeps API compatibility."""

    def __init__(self, *_, **__):
        pass

    def deliver(self, _html: str, _subject: str, _to: str) -> dict:
        raise RuntimeError("Delivery unavailable: openclaw delivery module is not installed/available")



_ISSUE_NO_LOCK = threading.Lock()


class SuperAgent:
    """실서비스 가능한 통합 오케스트레이터."""

    def __init__(
        self,
        root: Path,
        needs_file: Path | None = None,
        template_path: Path | None = None,
        use_browser_relay: bool = False,
    ):
        # execution policy: use_browser_relay=True -> BrowserRelayAdapter(실제 브라우저 기반),
        # False -> HttpNewsAdapter(요청 기반, 브라우저 비의존 모드)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.needs_file = needs_file or (self.root / "data" / "users" / "users.json")
        self.template_path = template_path or (self.root / "templates" / "CLUE_TEMPLATE_OFFICIAL.html")
        self.logger_root = self.root / "logs"
        self.logger_root.mkdir(parents=True, exist_ok=True)
        self.trace_enabled = os.getenv("CLUE_TRACE", "").lower() in {"1", "true", "yes", "on"}
        self.trace_root = Path(os.getenv("CLUE_TRACE_DIR", self.logger_root / "trace"))

        self.needs = NeedsAgent(self.needs_file)

        if use_browser_relay:
            adapter = BrowserRelayAdapter()
            self.adapter = adapter
            core_urls = self._load_core_feed_urls()

            def _core_search(q: str, limit: int = 25):
                return adapter.search_core_rss(
                    q,
                    core_urls,
                    limit=limit,
                    per_feed_limit=40,
                    total_limit=max(60, limit * 3),
                )

            self.collection = CollectionAgent(
                data_root=self.root / "data",
                needs_agent=self.needs,
                browser_search=adapter.search,
                fetch_body=adapter.fetch,
                resolve_url=adapter.resolve_google_news_url,
                search_core=_core_search,
                search_google=adapter.search_google_news if hasattr(adapter, "search_google_news") else adapter.search,
                min_count=8,
                daily_news_target=_env_int("CLUE_DAILY_NEWS_TARGET", 15),
            )
            self.collection_mode = "browser"
            self.collection_mode_reason = "브라우저 릴레이 사용"
        else:
            adapter = HttpNewsAdapter()
            self.adapter = adapter
            core_urls = self._load_core_feed_urls()

            def _core_search(q: str, limit: int = 25):
                return adapter.search_core_rss(
                    q,
                    core_urls,
                    limit=limit,
                    per_feed_limit=40,
                    total_limit=max(60, limit * 3),
                )

            self.collection = CollectionAgent(
                data_root=self.root / "data",
                needs_agent=self.needs,
                browser_search=None,
                fetch_body=adapter.fetch,
                resolve_url=adapter.resolve_google_news_url,
                search_core=_core_search,
                search_google=adapter.search_google_news if hasattr(adapter, "search_google_news") else adapter.search,
                min_count=8,
                daily_news_target=_env_int("CLUE_DAILY_NEWS_TARGET", 15),
            )
            self.collection_mode = "http"
            self.collection_mode_reason = "브라우저 미사용(HTTP) 모드"

        self.writer = WritingAgent(template_path=self.template_path, log_dir=self.logger_root, fetch_body=adapter.fetch, resolve_url=getattr(adapter, "resolve_google_news_url", None))
        self.delivery = self._build_delivery_manager()
        self._emit_delivery_warning_if_needed()

    def _trace_path(self, user: UserProfile, name: str) -> Path:
        return self.trace_root / str((user.name or "user").replace(" ", "_")).replace("..", "_") / "super" / name

    def _append_trace(self, path: Path, payload: dict) -> None:
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _emit_stage(self, user: UserProfile, issue_number: int | None, stage_id: str, status: str, *, in_count: int = 0, out_count: int = 0, elapsed_ms: int | None = None, extra: dict | None = None) -> None:
        if not self.trace_enabled:
            return
        rec = {
            "ts": datetime.now().astimezone().isoformat(),
            "component": "super",
            "stage_id": stage_id,
            "status": status,
            "user_code": user.user_code,
            "issue": int(issue_number or 0),
            "in_count": in_count,
            "out_count": out_count,
            "elapsed_ms": elapsed_ms,
            "extra": extra or {},
        }
        self._append_trace(self._trace_path(user, "stage_runs.jsonl"), rec)
        print(
            f"[super_stage] user={user.user_code} issue={int(issue_number or 0)} stage={stage_id} status={status} in={in_count} out={out_count}" +
            (f" elapsed_ms={elapsed_ms}" if elapsed_ms is not None else ""),
            flush=True,
        )

    def _log_collect_context(self, user: UserProfile):
        if self.collection_mode and self.collection_mode_reason:
            self._trace(f"collection_mode={self.collection_mode} reason={self.collection_mode_reason}")

    def _trace(self, message: str) -> None:
        if self.root:
            # 현재 구조에서는 stdout으로 로그만 출력
            print(f"[clue_letter] {message}", flush=True)

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


    def _build_delivery_manager(self):
        mode = _delivery_mode()
        if _HAS_OPENCLAW_DELIVERY and _OpenclawDeliveryManager is not None:
            return _OpenclawDeliveryManager()
        if mode == "openclaw-strict":
            raise RuntimeError(
                "Openclaw delivery is required for this run, but openclaw.agents.delivery is unavailable."
            )
        return LocalFileDelivery(self.logger_root / "delivery")

    def _latest_output_html(self, user: UserProfile, user_dir: Path, issue: str | None = None) -> Path:
        if not user_dir.exists():
            raise FileNotFoundError(f"user output directory missing: {user_dir}")

        candidates = []
        if issue:
            direct = user_dir / f"{issue}_{user.user_code}.html"
            if direct.exists():
                return direct
        for p in user_dir.glob(f"*_{user.user_code}.html"):
            if p.is_file() and p.suffix.lower() == ".html":
                candidates.append(p)
        if not candidates:
            raise FileNotFoundError(f"no rendered html found for user: {user.user_code}")

        def _sort_key(pp: Path) -> tuple[int, float]:
            stem = pp.stem
            date_part = stem.split("_", 1)[0] if "_" in stem else "00000000"
            try:
                return (int(date_part), pp.stat().st_mtime)
            except Exception:
                return (0, pp.stat().st_mtime)

        candidates.sort(key=_sort_key)
        return candidates[-1]

    def _load_latest_issue_no(self, user_dir: Path, html_path: Path) -> int:
        issue_file = user_dir / "issue_no.txt"
        if issue_file.exists():
            try:
                return int((issue_file.read_text(encoding="utf-8") or "0").strip())
            except Exception:
                pass

        stem = html_path.stem
        return 0

    def _parse_html_issue(self, html_path: Path, default_issue: int = 0) -> int:
        stem = html_path.stem
        date_part = stem.split("_", 1)[0]
        if date_part.isdigit() and len(date_part) == 8:
            return int(date_part)
        return default_issue

    def _send_existing_html(self, user: UserProfile, html_path: Path, issue_no: int):
        t2 = time.perf_counter()
        delivery = self.delivery.deliver(html_path.read_text(encoding="utf-8"), f"{datetime_stamp()}", user.email)
        t3 = time.perf_counter()
        if self.trace_enabled:
            self._emit_stage(user, issue_no, stage_id="delivery", status="done", in_count=1, out_count=1, elapsed_ms=int((t3 - t2) * 1000), extra={"delivery_mode": self._delivery_info().get("mode")})
        return {
            "ok": bool(delivery.get("ok", False)),
            "mode": "mail",
            "user_code": user.user_code,
            "issue_no": issue_no,
            "html_path": str(html_path),
            "email": {**delivery, **self._delivery_info(), "ok": bool(delivery.get("ok", True))},
            "timing": {
                "collect_sec": 0.0,
                "write_sec": 0.0,
                "delivery_sec": round(t3 - t2, 2),
                "total_sec": round(t3 - t2, 2),
            },
        }

    def _delivery_info(self) -> dict:
        available = bool(_HAS_OPENCLAW_DELIVERY)
        mode = _delivery_mode()
        if available:
            effective = "openclaw"
        elif mode == "fallback":
            effective = "local-fallback"
        elif mode == "openclaw-strict":
            effective = "openclaw-required"
        else:
            # Compatibility mode (gog/openclaw) with fallback delivery.
            effective = "openclaw-compatible"
        return {
            "mode": effective,
            "available": available,
            "policy": mode,
        }

    def _emit_delivery_warning_if_needed(self) -> None:
        if _HAS_OPENCLAW_DELIVERY:
            return
        if _delivery_mode() != "openclaw-strict":
            self._trace("openclaw.agents.delivery import failed; using local-file delivery fallback")

    def _load_core_feed_urls(self) -> list[str]:
        candidates = []
        active_cfg = self.root / "core_rss_active.json"
        legacy_cfg = self.root / "core_rss_registered.json"
        legacy_md = self.root / "core_rss_active.md"

        sources = None
        if active_cfg.exists():
            try:
                data = json.loads(active_cfg.read_text(encoding="utf-8"))
                sources = [item.get("url") for item in data.get("active", []) if item.get("url")]
            except Exception:
                sources = None

        if sources is None and legacy_cfg.exists():
            try:
                data = json.loads(legacy_cfg.read_text(encoding="utf-8"))
                sources = [item.get("url") for item in data.get("active", []) if item.get("url")]
            except Exception:
                sources = None

        if not sources and legacy_md.exists():
            for line in legacy_md.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("-") and "http" in line:
                    url = line.split(" ")[-1]
                    if url.startswith("http"):
                        sources.append(url)

        return sources or []
    def _next_issue_no(self, user_code: str) -> int:
        with _ISSUE_NO_LOCK:
            log = self.root / "logs" / "issue_no_tracker.json"
            if not log.exists():
                data = {}
            else:
                data = json.loads(log.read_text(encoding="utf-8"))
            current = int(data.get(user_code, 0)) + 1
            data[user_code] = current
            log.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return current

    def run_for_user(self, user_code: str, dry_run: bool = False, min_count: int = 8, send: bool = True, send_only: bool = False, issue: str | None = None):
        user = self.needs.get_user(user_code)
        if not user.is_active:
            raise RuntimeError(f"user not active: {user_code}")

        # send-only: use pre-rendered html and skip collection/writing.
        if send_only:
            out_dir = self.logger_root / str((user.name or "user").replace(" ", "_").replace("..", "_")) / "outputs"
            html_path = self._latest_output_html(user, out_dir, issue=issue)
            issue_no = self._parse_html_issue(html_path, self._next_issue_no(user.user_code))
            if self.trace_enabled:
                self._emit_stage(user, issue_no, stage_id="orchestrate", status="start", in_count=1, out_count=0)
            if self.trace_enabled:
                self._emit_stage(user, issue_no, stage_id="delivery", status="start", in_count=1, out_count=0)

            if not send:
                return {
                    "ok": True,
                    "mode": "send-only",
                    "user_code": user.user_code,
                    "issue_no": issue_no,
                    "html_path": str(html_path),
                    "email": {"ok": True, "mode": "delivery-skipped", "note": "send-only requires send path"},
                    "timing": {
                        "collect_sec": 0.0,
                        "write_sec": 0.0,
                        "delivery_sec": 0.0,
                        "total_sec": 0.0,
                    },
                }

            try:
                out = self._send_existing_html(user, html_path, issue_no)
            except Exception as e:
                return {
                    "ok": False,
                    "mode": "mail-error",
                    "user_code": user.user_code,
                    "issue_no": issue_no,
                    "html_path": str(html_path),
                    "email": {
                        "ok": False,
                        "mode": "delivery-error",
                        "error": str(e),
                        "note": "run failed at delivery stage",
                    },
                    "timing": {
                        "collect_sec": 0.0,
                        "write_sec": 0.0,
                        "delivery_sec": 0.0,
                        "total_sec": 0.0,
                    },
                }
            if self.trace_enabled:
                self._emit_stage(user, issue_no, stage_id="orchestrate", status="done", in_count=1, out_count=1, elapsed_ms=int(float(out["timing"]["total_sec"]) * 1000), extra={"collect_sec": 0.0, "write_sec": 0.0})
            return out

        t0 = time.perf_counter()
        self._log_collect_context(user)
        if self.trace_enabled:
            self._emit_stage(user, None, stage_id="orchestrate", status="start", in_count=1, out_count=0)
        if self.trace_enabled:
            self._emit_stage(user, None, stage_id="collect", status="start", in_count=1, out_count=0)
        collected = self.collection.collect(user=user, min_count=min_count)
        t1 = time.perf_counter()
        if self.trace_enabled:
            self._emit_stage(user, None, stage_id="collect", status="done", in_count=1, out_count=len(collected), elapsed_ms=int((t1 - t0) * 1000))
        # writing 단계는 수집된 daily letter 파일을 user 단위로 다시 읽어 작성한다.
        load_t0 = time.perf_counter()
        articles = self.collection.load_daily_news(user=user)
        if not articles:
            articles = collected
        t_load = time.perf_counter()
        issue_no = self._next_issue_no(user.user_code)
        if self.trace_enabled:
            self._emit_stage(user, issue_no, stage_id="load_daily", status="done", in_count=len(collected), out_count=len(articles), elapsed_ms=int((t_load - load_t0) * 1000))
        if self.trace_enabled:
            self._emit_stage(user, issue_no, stage_id="write", status="start", in_count=len(articles), out_count=0)
        html_path = self.writer.build_and_save(user=user, collected=articles, issue_number=issue_no)
        t2 = time.perf_counter()
        if self.trace_enabled:
            self._emit_stage(user, issue_no, stage_id="write", status="done", in_count=len(articles), out_count=1, elapsed_ms=int((t2 - t_load) * 1000))

        if self.trace_enabled:
            self._emit_stage(user, issue_no, stage_id="orchestrate", status="done", in_count=len(articles), out_count=1, elapsed_ms=int((t2 - t0) * 1000), extra={"collect_sec": round(t1 - t0, 2), "write_sec": round(t2 - t1, 2)})

        if dry_run:
            out = {
                "ok": True,
                "mode": "dry-run",
                "user_code": user.user_code,
                "issue_no": issue_no,
                "article_count": len(articles),
                "html_path": str(html_path),
                "timing": {
                    "collect_sec": round(t1 - t0, 2),
                    "write_sec": round(t2 - t1, 2),
                    "total_sec": round(t2 - t0, 2),
                },
            }
            if not _HAS_OPENCLAW_DELIVERY:
                out["delivery_fallback"] = self._delivery_info()
            return out

        if not send:
            return {
                "ok": True,
                "mode": "generated",
                "user_code": user.user_code,
                "issue_no": issue_no,
                "article_count": len(articles),
                "html_path": str(html_path),
                "email": {"ok": True, "mode": "skipped", "note": "delivery skipped by send=False"},
                "timing": {
                    "collect_sec": round(t1 - t0, 2),
                    "write_sec": round(t2 - t1, 2),
                    "total_sec": round(t2 - t0, 2),
                },
            }

        if self.trace_enabled:
            self._emit_stage(user, issue_no, stage_id="delivery", status="start", in_count=1, out_count=0)
        try:
            delivery = self.delivery.deliver(html_path.read_text(encoding="utf-8"), f"{datetime_stamp()}", user.email)
            delivery = {**delivery, **self._delivery_info(), "ok": bool(delivery.get("ok", True))}
            mode = "mail"
        except Exception as e:
            delivery = {
                "ok": False,
                "mode": "delivery-error",
                "error": str(e),
                "note": "run failed at delivery stage",
            }
            mode = "mail-error"
        t3 = time.perf_counter()
        if self.trace_enabled:
            self._emit_stage(user, issue_no, stage_id="delivery", status="done", in_count=1, out_count=1, elapsed_ms=int((t3 - t2) * 1000), extra={"delivery_mode": self._delivery_info().get("mode")})
        return {
            "ok": bool(delivery.get("ok", False)),
            "mode": mode,
            "user_code": user.user_code,
            "issue_no": issue_no,
            "article_count": len(articles),
            "html_path": str(html_path),
            "email": delivery,
            "timing": {
                "collect_sec": round(t1 - t0, 2),
                "write_sec": round(t2 - t1, 2),
                "delivery_sec": round(t3 - t2, 2),
                "total_sec": round(t3 - t0, 2),
            },
        }

    def run_all(self, dry_run: bool = False, min_count: int = 8, send: bool = True):
        out = []
        for user in self.needs.list_users():
            if not user.is_active:
                continue
            try:
                out.append(self.run_for_user(user.user_code, dry_run=dry_run, min_count=min_count, send=send))
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
