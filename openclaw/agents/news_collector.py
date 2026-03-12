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

    def _domain_of(self, url: str) -> str:
        try:
            d = (urlparse((url or '').strip()).netloc or '').lower()
            return self._normalize_domain(d)
        except Exception:
            return ''

    def _domain_trust_weight(self, domain: str) -> int:
        pf = self.sources_cfg.get("pre_filters", {}) if isinstance(self.sources_cfg, dict) else {}
        weights = pf.get("domainTrustWeights", {}) if isinstance(pf, dict) else {}
        if not isinstance(weights, dict):
            return 0
        d = self._normalize_domain(domain)
        for k, v in weights.items():
            kd = self._normalize_domain(str(k))
            try:
                iv = int(v)
            except Exception:
                iv = 0
            if d == kd or d.endswith(f".{kd}"):
                return iv
        return 0

    def _extractability_penalty(self, domain: str) -> int:
        pf = self.sources_cfg.get("pre_filters", {}) if isinstance(self.sources_cfg, dict) else {}
        arr = pf.get("lowExtractabilityPenaltyDomains", []) if isinstance(pf, dict) else []
        try:
            pen = int(pf.get("lowExtractabilityPenalty", 2))
        except Exception:
            pen = 2
        d = self._normalize_domain(domain)
        for x in arr or []:
            xd = self._normalize_domain(str(x))
            if d == xd or d.endswith(f".{xd}"):
                return pen
        return 0

    def _domain_max_share(self) -> float:
        pf = self.sources_cfg.get("pre_filters", {}) if isinstance(self.sources_cfg, dict) else {}
        try:
            v = float(pf.get("domainMaxShare", 0.30))
        except Exception:
            v = 0.30
        if v <= 0:
            return 0.30
        if v > 1:
            return 1.0
        return v

    def _rank_and_limit(self, articles: list[dict], keywords: list[str], limit: int) -> list[dict]:
        seen = set()
        uniq = []
        for a in articles:
            u = (a.get("url") or "").strip().lower()
            if not u or u in seen:
                continue
            seen.add(u)

            domain = self._domain_of(u)
            base = self._score(a, keywords)
            if a.get("source") == "google_news":
                base -= 2

            # 원문 읽기 전 프리필터 가중치: 도메인 신뢰도(+), 추출난이도(-)
            base += self._domain_trust_weight(domain)
            base -= self._extractability_penalty(domain)

            a["_score"] = base
            a["_domain"] = domain
            uniq.append(a)

        uniq.sort(key=lambda x: (x.get("_score", 0), x.get("published_at", "")), reverse=True)

        # 편중 방지: 단일 도메인 상한
        max_share = self._domain_max_share()
        max_per_domain = max(1, int(limit * max_share))
        out = []
        counts = {}
        for a in uniq:
            d = a.get("_domain", "")
            if d:
                if counts.get(d, 0) >= max_per_domain:
                    continue
                counts[d] = counts.get(d, 0) + 1
            out.append(a)
            if len(out) >= limit:
                break

        # cleanup internal fields
        for a in out:
            a.pop("_domain", None)
        return out

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
