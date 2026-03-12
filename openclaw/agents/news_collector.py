from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import requests
from tenacity import retry, stop_after_attempt, wait_exponential


class NewsCollector:
    def __init__(self, sources_cfg: dict, freshness_hours: int = 24, forbidden_domains: set[str] | None = None):
        self.sources_cfg = sources_cfg
        self.freshness_hours = freshness_hours
        self.forbidden_domains = {self._normalize_domain(d) for d in (forbidden_domains or set()) if d}

    def collect_all(self, per_category_limit: int = 30) -> list[dict]:
        pool = []
        categories = self.sources_cfg.get("categories", {})
        for category, cfg in categories.items():
            items = []
            tasks = []
            with ThreadPoolExecutor(max_workers=10) as ex:
                for url in cfg.get("rss", []):
                    tasks.append(ex.submit(self._fetch_rss, url, category))
                for q in cfg.get("google_news_queries", []):
                    tasks.append(ex.submit(self._fetch_google_news, q, category))

                for fut in as_completed(tasks):
                    try:
                        items.extend(fut.result() or [])
                    except Exception:
                        continue

            ranked = self._rank_and_limit(items, cfg.get("keywords", []), per_category_limit)
            pool.extend(ranked)
        return self._dedup_pool(pool)

    def collect_custom_queries(self, queries: list[str], category: str = "AI_TECH", limit: int = 40) -> list[dict]:
        if not queries:
            return []
        items = []
        regions = [("US", "en"), ("KR", "ko"), ("TW", "zh-TW"), ("CN", "zh-CN")]

        tasks = []
        with ThreadPoolExecutor(max_workers=12) as ex:
            for q in queries:
                if not q:
                    continue
                for gl, hl in regions:
                    tasks.append(ex.submit(self._fetch_google_news, q, category, gl, hl))

            for fut in as_completed(tasks):
                try:
                    items.extend(fut.result() or [])
                except Exception:
                    continue

        ranked = self._rank_and_limit(items, keywords=[], limit=limit)
        return self._dedup_pool(ranked)

    def _normalize_domain(self, domain: str) -> str:
        d = (domain or "").strip().lower()
        if d.startswith("www."):
            d = d[4:]
        return d

    def _is_forbidden_url(self, url: str) -> bool:
        u = (url or "").strip().lower()
        if not u:
            return False
        host = self._normalize_domain((urlparse(u).netloc or "").lower())
        for d in self.forbidden_domains:
            if host == d or host.endswith(f".{d}") or d in u:
                return True
        return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def _fetch_rss(self, url: str, category: str) -> list[dict]:
        if self._is_forbidden_url(url):
            return []
        feed = feedparser.parse(url)
        return self._normalize_entries(feed.entries, source=url, category=category)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def _fetch_google_news(self, query: str, category: str, gl: str = "US", hl: str = "en") -> list[dict]:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={gl}:{hl}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        feed = feedparser.parse(r.text)
        return self._normalize_entries(feed.entries, source="google_news", category=category)

    def _normalize_entries(self, entries, source: str, category: str) -> list[dict]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.freshness_hours)
        out = []
        for e in entries:
            pub = self._parse_date(e)
            if pub and pub < cutoff:
                continue
            raw_url = getattr(e, "link", "").strip()
            resolved_url = self._resolve_source_url(raw_url)
            if self._is_forbidden_url(resolved_url):
                continue
            out.append(
                {
                    "title": getattr(e, "title", "").strip(),
                    "summary": getattr(e, "summary", "").strip(),
                    "url": resolved_url,
                    "source": source,
                    "source_category": category,
                    "published_at": pub.isoformat() if pub else "",
                }
            )
        return out

    @staticmethod
    def _parse_date(entry):
        for k in ("published", "updated"):
            v = getattr(entry, k, None)
            if v:
                try:
                    dt = parsedate_to_datetime(v)
                    return dt.astimezone(timezone.utc)
                except Exception:
                    pass
        return None

    def _resolve_source_url(self, url: str) -> str:
        u = (url or "").strip()
        if not u:
            return ""
        if "news.google.com" not in u:
            return self._canonicalize_url(u)

        try:
            parsed = urlparse(u)
            qs = parse_qs(parsed.query)
            if qs.get("url") and qs["url"][0]:
                return self._canonicalize_url(qs["url"][0])

            # google rss article redirect는 최종 URL 추적
            r = requests.get(u, timeout=10, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
            final = (r.url or "").strip()
            if final and "news.google.com" not in final:
                return self._canonicalize_url(final)
        except Exception:
            pass

        return self._canonicalize_url(u)

    @staticmethod
    def _canonicalize_url(url: str) -> str:
        try:
            p = urlparse((url or "").strip())
            if not p.scheme or not p.netloc:
                return (url or "").strip()
            # 추적성 파라미터 제거
            drop_prefix = ("utm_", "fbclid", "gclid", "oc", "ref", "ref_src")
            q = parse_qs(p.query, keep_blank_values=False)
            kept = []
            for k, vals in q.items():
                lk = k.lower()
                if lk.startswith(drop_prefix) or lk in drop_prefix:
                    continue
                for v in vals:
                    kept.append((k, v))
            from urllib.parse import urlencode
            query = urlencode(kept, doseq=True)
            clean = p._replace(query=query, fragment="")
            return clean.geturl().rstrip("?")
        except Exception:
            return (url or "").strip()

    @staticmethod
    def _score(item: dict, keywords: list[str]) -> int:
        text = f"{item.get('title','')} {item.get('summary','')}".lower()
        return sum(text.count(k.lower()) for k in keywords)

    def _rank_and_limit(self, articles: list[dict], keywords: list[str], limit: int) -> list[dict]:
        seen = set()
        uniq = []
        for a in articles:
            u = (a.get("url") or "").strip().lower()
            if not u or u in seen:
                continue
            seen.add(u)
            base = self._score(a, keywords)
            if a.get("source") == "google_news":
                base -= 2
            a["_score"] = base
            uniq.append(a)
        uniq.sort(key=lambda x: (x.get("_score", 0), x.get("published_at", "")), reverse=True)
        return uniq[:limit]

    @staticmethod
    def _dedup_pool(pool: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for a in pool:
            key = (a.get("url", "").strip().lower(), a.get("title", "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(a)
        return out
