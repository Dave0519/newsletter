from __future__ import annotations

import json
import re
import html
import subprocess
import os
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import quote_plus, unquote, urlparse, parse_qs

try:
    import requests
except Exception:
    requests = None
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from googlenewsdecoder import gnewsdecoder as _gnewsdecoder
except Exception:  # optional dependency
    _gnewsdecoder = None


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    source: str = "browser"
    published_at: str = ""


def _run_browser_cmd(*args: str, timeout_ms: int = 90000) -> str:
    cmd = ["openclaw", "browser", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_ms / 1000)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"openclaw browser command timeout after {timeout_ms}ms: {' '.join(cmd)}")

    if proc.returncode != 0:
        raise RuntimeError(f"openclaw browser command failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout.strip()


def _decode_ddg_url(raw: str) -> str:
    try:
        p = urlparse(raw)
        if p.netloc.endswith("duckduckgo.com") and p.path.startswith("/l/"):
            qs = parse_qs(p.query)
            dst = (qs.get("uddg") or [""])[0]
            if dst:
                return unquote(dst)
    except Exception:
        return raw
    return raw


def _json_load_last(s: str) -> Any:
    s = s.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        start = s.rfind("{")
        if start == -1:
            raise
        end = s.rfind("}")
        if end > start:
            return json.loads(s[start : end + 1])
        raise


def _dedupe_keep_order(items: list[SearchHit]) -> list[SearchHit]:
    out: list[SearchHit] = []
    seen = set()
    for it in items:
        if it.url in seen:
            continue
        seen.add(it.url)
        out.append(it)
    return out


def _append_search_trace(source: str, query: str, **extra) -> None:
    """RSS/Google 검색 키워드 및 타겟 로그를 jsonl로 남긴다."""
    try:
        root = Path(__file__).resolve().parents[1]  # repository root/agents
        log_root = root / "logs" / "search_trace"
        log_root.mkdir(parents=True, exist_ok=True)
        day = datetime.now().strftime("%Y-%m-%d")
        path = log_root / f"{day}.jsonl"
        rec = {
            "ts": datetime.now().isoformat(),
            "source": source,
            "query": (query or "").strip(),
        }
        rec.update(extra or {})
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        return


class BrowserRelayAdapter:
    def __init__(self, search_engine: str = "duckduckgo", open_timeout_ms: int = 20000):
        self.search_engine = search_engine
        self.open_timeout_ms = open_timeout_ms
        self._last_target: Optional[str] = None
        self._started = False
        # 피드 원문은 쿼리별 재사용을 위해 캐시(동일 실행 내 반복 오픈 최소화)
        self._rss_cache: dict[str, tuple[float, str]] = {}
        self._rss_cache_ttl = 180.0

    def _ensure_start(self):
        if self._started:
            return

        # start may take longer than default timeout on first launch; probe status first,
        # then launch with a generous timeout and recover from transient waits.
        try:
            _run_browser_cmd("status", timeout_ms=5000)
            self._started = True
            return
        except Exception:
            pass

        try:
            _run_browser_cmd("start", timeout_ms=180000)
            self._started = True
            return
        except Exception as e:
            # fallback: if start timed out but gateway/browser booted, continue
            try:
                _run_browser_cmd("status", timeout_ms=10000)
                self._started = True
                return
            except Exception:
                raise

    def _refresh_last_target(self) -> Optional[str]:
        try:
            raw = _run_browser_cmd("tabs", "--json")
            data = _json_load_last(raw)
            if isinstance(data, dict) and "tabs" in data:
                tabs = data.get("tabs")
            else:
                tabs = data
            if isinstance(tabs, list) and tabs:
                for t in tabs:
                    if isinstance(t, dict) and t.get("targetId"):
                        self._last_target = t.get("targetId")
                        return self._last_target
        except Exception:
            return self._last_target
        return self._last_target

    def _wait_load(self, target_id: str | None, wait_state: str = "domcontentloaded", timeout_ms: int = 12000) -> None:
        if not target_id:
            return
        try:
            _run_browser_cmd(
                "wait",
                "--load",
                wait_state,
                "--target-id",
                str(target_id),
                "--timeout-ms",
                str(timeout_ms),
                timeout_ms=timeout_ms + 2000,
            )
        except Exception:
            # fallback to short generic wait to avoid hard blocking
            try:
                _run_browser_cmd("wait", "--time", str(min(timeout_ms, 2500)), timeout_ms=timeout_ms)
            except Exception:
                pass

    def _run_with_target(self, script: str, context: str, max_retry: int = 2, eval_timeout_ms: int = 18000) -> Any:
        self._ensure_start()
        if not self._last_target:
            self._refresh_last_target()

        if self._last_target:
            try:
                _run_browser_cmd("focus", self._last_target)
            except Exception:
                self._last_target = None

        if self._last_target:
            self._wait_load(self._last_target, "domcontentloaded", timeout_ms=8000)

        last_err: Exception | None = None
        for attempt in range(max_retry + 1):
            try:
                if attempt > 0 and self._last_target:
                    try:
                        self._wait_load(self._last_target, "networkidle", timeout_ms=10000)
                    except Exception:
                        pass

                out = _run_browser_cmd("evaluate", "--fn", script, "--json", timeout_ms=eval_timeout_ms)
                return _json_load_last(out)
            except Exception as e:
                last_err = e
                msg = str(e).lower()

                if "tab not found" in msg or "target" in msg and "not found" in msg:
                    self._refresh_last_target()
                    if self._last_target:
                        try:
                            _run_browser_cmd("focus", self._last_target)
                        except Exception:
                            pass
                    if attempt < max_retry:
                        continue

                if "gateway timeout" in msg or "openclaw browser command timeout" in msg or "timed out" in msg:
                    time.sleep(1)
                    if attempt < max_retry:
                        continue

                if attempt >= max_retry:
                    raise

                # short pause between retries for slower pages
                time.sleep(0.6)

        if last_err:
            raise last_err
        raise RuntimeError(f"browser evaluate failed for {context}")

    def _open(self, url: str) -> str:
        self._ensure_start()
        out = _run_browser_cmd("open", url, "--json", timeout_ms=self.open_timeout_ms)
        data = _json_load_last(out)
        if isinstance(data, dict):
            tid = data.get("targetId")
            if tid:
                self._last_target = tid
                self._wait_load(tid, "domcontentloaded", timeout_ms=8000)
        return out

    def _is_candidate_article_url(self, url: str) -> bool:
        if not url:
            return False
        u = (url or "").strip().lower()
        if not u.startswith(("http://", "https://")):
            return False
        if u == "#" or u.startswith("javascript:") or u.startswith("mailto:") or u.startswith("tel:"):
            return False
        return True

    def search(self, query: str, limit: int = 20, min_text_len: int = 12) -> list[SearchHit]:
        q = quote_plus(query)
        search_url = f"https://duckduckgo.com/html/?q={q}"

        self._open(search_url)
        if not self._last_target:
            self._refresh_last_target()

        script = (
            "() => Array.from(document.querySelectorAll('a'))"
            ".map(a => ({\"text\": (a.innerText||'').trim(), \"href\": (a.href||'').trim()}))"
            f".filter(x => x.text && x.text.length > {min_text_len} && x.href && x.href.startsWith('http'))"
            ".slice(0, 220)"
        )

        data = self._run_with_target(script, context="search", max_retry=2, eval_timeout_ms=10000)
        nodes = data.get("result") if isinstance(data, dict) else data

        hits: list[SearchHit] = []
        if isinstance(nodes, list):
            for item in nodes:
                title = (item.get("text") or "").strip()
                href = _decode_ddg_url(item.get("href") or "")
                if not title or not href:
                    continue
                low = href.lower()
                if low.startswith(("#", "javascript:", "mailto:", "tel:")):
                    continue
                if "duckduckgo.com/html/?q=" in low:
                    continue
                if len(title) < min_text_len:
                    continue
                hits.append(SearchHit(title=title, url=href, source=self.search_engine))
                if len(hits) >= limit * 2:
                    break

        return _dedupe_keep_order(hits)[:limit]




    def resolve_google_news_url(self, url: str) -> str | None:
        """Google News RSS 링크(news.google.com/rss/articles/...)를 원문 URL로 변환."""
        self._open(url)
        if not self._last_target:
            self._refresh_last_target()

        script = r"""() => ({
            href: window.location.href || '',
            canonical: (() => {
                const sel = document.querySelector('link[rel="canonical"]');
                return sel ? (sel.href || sel.getAttribute('href') || '') : '';
            })(),
            ogUrl: (() => {
                const sel = document.querySelector('meta[property="og:url"]') || document.querySelector('meta[name="twitter:url"]');
                return sel ? (sel.content || '') : '';
            })()
        })"""
        data = self._run_with_target(script, context="resolve_google_news", max_retry=1, eval_timeout_ms=10000)
        if not isinstance(data, dict):
            return None
        result = data.get("result") or {}
        if not isinstance(result, dict):
            return None
        for key in ("canonical", "ogUrl", "href"):
            candidate = (result.get(key) or "").strip()
            if candidate.startswith("http"):
                return candidate
        return None


    def _extract_google_titles(self, text: str, query: str, limit: int = 50) -> list[dict[str, str]]:
        if not text:
            return []

        def _clean_xml(v: str) -> str:
            return re.sub(r"<!\[CDATA\[(.*?)\]>", lambda m: m.group(1), v or "", flags=re.S)

        out: list[dict[str, str]] = []
        query_low = (query or "").lower()
        tokens = [x for x in re.split(r"\s+", query_low) if x and len(x) >= 2]

        import xml.etree.ElementTree as ET
        root = None
        try:
            root = ET.fromstring(text)
        except Exception:
            root = None

        if root is not None:
            for item in root.findall(".//item")[: limit * 5]:
                title = _clean_xml((item.findtext("title") or ""))
                link = _clean_xml((item.findtext("link") or ""))
                desc = _clean_xml((item.findtext("description") or ""))
                pub = _clean_xml((item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date") or ""))
                if not title or not link:
                    continue
                if not self._is_candidate_article_url(link):
                    continue

                blob = f"{title} {desc}".lower()
                if query_low and query_low not in blob and not any(tok in blob for tok in tokens):
                    continue

                out.append({
                    "title": re.sub(r"\s+", " ", title).strip(),
                    "url": re.sub(r"\s+", " ", link).strip(),
                    "snippet": re.sub(r"\s+", " ", desc).strip(),
                    "source": "rss",
                    "published_at": re.sub(r"\s+", " ", pub).strip(),
                })

        if not out:
            # fallback regex parser
            for item in re.findall(r"<item>(.*?)</item>", text, flags=re.S | re.I):
                title = re.search(r"<title[^>]*>(.*?)</title>", item, flags=re.S | re.I)
                link = re.search(r"<link[^>]*>(.*?)</link>", item, flags=re.S | re.I)
                desc = re.search(r"<description[^>]*>(.*?)</description>", item, flags=re.S | re.I)
                pub = re.search(r"<pubDate[^>]*>(.*?)</pubDate>", item, flags=re.S | re.I)
                dt = re.search(r"<dc:date[^>]*>(.*?)</dc:date>", item, flags=re.S | re.I)
                if not title or not link:
                    continue
                t = _clean_xml(title.group(1))
                l = _clean_xml(link.group(1))
                d = _clean_xml(desc.group(1) if desc else "")
                p = _clean_xml(pub.group(1) if pub else (dt.group(1) if dt else ""))
                if not t or not l:
                    continue
                if not self._is_candidate_article_url(l):
                    continue
                blob = f"{t} {d}".lower()
                if query_low and query_low not in blob and not any(tok in blob for tok in tokens):
                    continue
                out.append({"title": re.sub(r"\s+", " ", t).strip(), "url": re.sub(r"\s+", " ", l).strip(), "snippet": re.sub(r"\s+", " ", d).strip(), "source": "rss", "published_at": re.sub(r"\s+", " ", p).strip()})

        uniq: list[dict[str, str]] = []
        seen = set()
        for item in out[: limit * 4]:
            u = (item.get("url") or "").strip()
            if not u or u in seen:
                continue
            seen.add(u)
            uniq.append(item)
            if len(uniq) >= limit:
                break
        return uniq


    def search_core_rss(self, query: str, feed_urls: list[str], limit: int = 20, per_feed_limit: int = 80, total_limit: int | None = None) -> list[SearchHit]:
        q = (query or "").strip()
        if not q or not feed_urls:
            return []

        all_hits: list[dict[str, str]] = []
        if total_limit is None:
            total_limit = limit * 5
        for feed_url in feed_urls[:20]:
            try:
                raw = self._fetch_feed_raw(feed_url, cache_ttl=self._rss_cache_ttl)
            except Exception:
                _append_search_trace("core-rss", q, feed_url=feed_url, status="fetch_error", hit_count=0)
                continue
            if not isinstance(raw, str) or not raw:
                _append_search_trace("core-rss", q, feed_url=feed_url, status="empty", hit_count=0)
                continue
            hits = self._extract_google_titles(raw, q, limit=per_feed_limit)
            _append_search_trace("core-rss", q, feed_url=feed_url, status="ok", hit_count=len(hits))
            all_hits.extend(hits)

        uniq: list[dict[str, str]] = []
        seen=set()
        for h in all_hits:
            url=h.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            uniq.append(h)
            if len(uniq)>=total_limit:
                break
        return [SearchHit(title=h["title"], url=h["url"], snippet=h["snippet"], source="core-rss", published_at=h.get("published_at", "")) for h in uniq[:total_limit]]


    def fetch(self, url: str) -> str:
        self._open(url)
        if not self._last_target:
            self._refresh_last_target()

        script = r"""() => ({
            title: document.title || '',
            body: (() => {
                const toText = (node) => (node && node.innerText || '').replace(/\s+/g,' ').trim();
                const candidates = [
                    document.querySelector('article'),
                    document.querySelector('main'),
                    document.querySelector('[role=\"main\"]'),
                    document.querySelector('.content'),
                    document.querySelector('.article'),
                    document.querySelector('.post-content'),
                    document.querySelector('.post-body'),
                    document.querySelector('.entry-content'),
                    document.body
                ];
                let text = '';
                for (let i = 0; i < candidates.length; i++) {
                    const c = candidates[i];
                    if (!c) continue;
                    const t = toText(c);
                    if (t && t.length > 260) { text = t; break; }
                }
                if (!text) {
                    const walker = document.createTreeWalker(document.body || document, NodeFilter.SHOW_TEXT, null, false);
                    let node;
                    const texts = [];
                    while ((node = walker.nextNode())) {
                        const t = (node.textContent || '').replace(/\s+/g,' ').trim();
                        if (t.length >= 20) texts.push(t);
                    }
                    text = texts.slice(0,1200).join('\\n');
                }
                return text.slice(0,10000).replace(/\\n{2,}/g, '\\n');
            })()
        })"""

        try:
            data = self._run_with_target(script, context="fetch", max_retry=1, eval_timeout_ms=14000)
        except Exception:
            return ""

        result = data.get("result") if isinstance(data, dict) else {}
        if isinstance(result, dict):
            body = (result.get("body") or "").strip()
            title = (result.get("title") or "").strip()
            if title and body:
                return f"{title}\n{body}"
            return body
        if isinstance(result, str):
            return re.sub(r"\s+", " ", result).strip()
        return ""


class HttpNewsAdapter:
    """브라우저 없이 HTTP/requests 기반 수집용 경량 어댑터."""

    def _extract_google_news_target(self, html_text: str) -> str | None:
        if not html_text:
            return None

        try:
            # 1) canonical / og / twitter URL
            if BeautifulSoup is not None:
                soup = BeautifulSoup(html_text or "", "html.parser")
                if soup is not None:
                    selectors = [
                        "link[rel='canonical']",
                        "meta[property='og:url']",
                        "meta[name='twitter:url']",
                    ]
                    for selector in selectors:
                        tag = soup.select_one(selector)
                        if not tag:
                            continue
                        cand = tag.get("href") if selector.startswith("link") else tag.get("content")
                        if cand and cand.startswith(("http://", "https://")):
                            return cand.strip()

                    # JSON-LD 또는 스크립트 블록에서 URL 필드 탐색
                    for st in soup.find_all("script"):
                        txt = (st.get_text() or "").strip()
                        if not txt:
                            continue
                        m = re.search(r'"url"\s*:\s*"(https:[^"]+)"', txt)
                        if m:
                            cand2 = (m.group(1) or "").strip()
                            if cand2.startswith(("http://", "https://")):
                                return cand2

            # 2) 텍스트 패턴 fallback
            pats = [
                r'canonical"\s*[:=]\s*"(https://[^"]+)"',
                r'og:url"\s*content="(https?://[^"]+)"',
                r'twitter:url"\s*content="(https?://[^"]+)"',
                r"https?://[^\"']*url=(https%3A%2F%2F[^&\"']+)",
            ]
            for pat in pats:
                m = re.search(pat, html_text or "")
                if not m:
                    continue
                g = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
                v = (g or "").strip('\"\' ')
                if v.startswith("https%3A%2F%2F"):
                    try:
                        v = unquote_plus(v)
                    except Exception:
                        pass
                if v.startswith(("http://", "https://")):
                    return v
            # 3) google 내부 인코딩 링크 추출
            for m in re.findall(r'"(https%3A%2F%2F[^"%]+(?:%2F[^"%]+)*)"', html_text or ""):
                try:
                    cand = unquote_plus(m)
                except Exception:
                    cand = m
                if cand.startswith(("http://", "https://")):
                    return cand
        except Exception:
            return None

        return None

    def __init__(self, timeout_ms: int = 12000, open_timeout_ms: int | None = None):
        self.timeout_ms = timeout_ms
        self.open_timeout_ms = open_timeout_ms or timeout_ms
        self._rss_cache: dict[str, tuple[float, str]] = {}
        self._rss_cache_ttl = 180.0

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        }

    def _http_get(self, url: str) -> str:
        if requests is None:
            raise RuntimeError("requests가 설치되지 않아 브라우저 없는 모드가 동작하지 않습니다.")
        resp = requests.get(url, timeout=self.timeout_ms / 1000, headers=self._headers(), allow_redirects=True)
        resp.raise_for_status()
        return resp.text or ""

    def _strip_text(self, html_text: str) -> str:
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html_text or "", "html.parser")
            return (soup.get_text(" ", strip=True) or "").replace(" ", " ")
        text = re.sub(r"<script[\s\S]*?</script>", " ", html_text or "", flags=re.I)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _is_candidate_article_url(self, url: str) -> bool:
        if not url:
            return False
        u = (url or "").strip().lower()
        if not u.startswith(("http://", "https://")):
            return False
        if u == "#" or u.startswith("javascript:") or u.startswith("mailto:") or u.startswith("tel:"):
            return False
        return True

    def _fetch_feed_raw(self, feed_url: str, cache_ttl: float = 180.0) -> str:
        now = time.time()
        entry = self._rss_cache.get(feed_url)
        if entry:
            ts, cached = entry
            if now - ts <= cache_ttl:
                return cached
        raw = self._http_get(feed_url)
        self._rss_cache[feed_url] = (now, raw)
        return raw

    def _extract_google_titles(self, text: str, query: str, limit: int = 50) -> list[dict[str, str]]:
        if not text:
            return []
        out: list[dict[str, str]] = []
        query_low = (query or "").lower()
        items = re.findall(r"<item>(.*?)</item>", text, flags=re.S | re.I)
        if items:
            tokens = [x for x in query_low.split() if x and len(x) >= 2]
            for item in items:
                title = re.search(r"<title[^>]*>(.*?)</title>", item, flags=re.S | re.I)
                link = re.search(r"<link[^>]*>(.*?)</link>", item, flags=re.S | re.I)
                desc = re.search(r"<description[^>]*>(.*?)</description>", item, flags=re.S | re.I)
                pub = re.search(r"<pubDate[^>]*>(.*?)</pubDate>", item, flags=re.S | re.I)
                dt = re.search(r"<dc:date[^>]*>(.*?)</dc:date>", item, flags=re.S | re.I)
                if not title or not link:
                    continue
                t = re.sub(r"\s+", " ", re.sub(r"<!\[CDATA\[(.*?)\]>", r"\1", title.group(1))).strip()
                if not t:
                    continue
                l = re.sub(r"\s+", " ", re.sub(r"<!\[CDATA\[(.*?)\]>", r"\1", link.group(1))).strip()
                d = re.sub(r"\s+", " ", re.sub(r"<!\[CDATA\[(.*?)\]>", r"\1", (desc.group(1) if desc else ""))).strip()
                p = re.sub(r"\s+", " ", re.sub(r"<!\[CDATA\[(.*?)\]>", r"\1", (pub.group(1) if pub else (dt.group(1) if dt else "")))).strip()
                blob = f"{t} {d}".lower()
                if not self._is_candidate_article_url(l):
                    continue
                if query_low and query_low not in blob and not any(tok in blob for tok in tokens):
                    continue
                out.append({"title": t, "url": l, "snippet": d, "source": "rss", "published_at": p})
        if not out:
            for link, txt in re.findall(r"<a[^>]+href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>", text, flags=re.S | re.I):
                t = re.sub(r"<[^>]+>", " ", re.sub(r"\s+", " ", txt)).strip()
                l = re.sub(r"<[^>]+>", " ", re.sub(r"\s+", " ", link)).strip()
                if not t or not l or not self._is_candidate_article_url(l):
                    continue
                tokens = [x for x in query_low.split() if x and len(x) >= 2]
                if query_low and query_low not in (t + " " + l).lower() and not any(tok in (t + " " + l).lower() for tok in tokens):
                    continue
                out.append({"title": t, "url": l, "snippet": "", "source": "rss", "published_at": ""})

        uniq: list[dict[str, str]] = []
        seen = set()
        for item in out[: limit * 4]:
            u = item.get("url")
            if not u or u in seen:
                continue
            seen.add(u)
            uniq.append(item)
            if len(uniq) >= limit:
                break
        return uniq

    def search_core_rss(self, query: str, feed_urls: list[str], limit: int = 20, per_feed_limit: int = 80, total_limit: int | None = None) -> list[SearchHit]:
        q = (query or "").strip()
        if not q or not feed_urls:
            return []
        all_hits: list[dict[str, str]] = []
        if total_limit is None:
            total_limit = limit * 5
        for feed_url in feed_urls[:20]:
            try:
                raw = self._fetch_feed_raw(feed_url, cache_ttl=self._rss_cache_ttl)
            except Exception:
                _append_search_trace("core-rss", q, feed_url=feed_url, status="fetch_error", hit_count=0)
                continue
            if not isinstance(raw, str) or not raw:
                _append_search_trace("core-rss", q, feed_url=feed_url, status="empty", hit_count=0)
                continue
            hits = self._extract_google_titles(raw, q, limit=per_feed_limit)
            _append_search_trace("core-rss", q, feed_url=feed_url, status="ok", hit_count=len(hits))
            all_hits.extend(hits)
        uniq: list[dict[str, str]] = []
        seen=set()
        for h in all_hits:
            url=h.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            uniq.append(h)
            if len(uniq)>=total_limit:
                break
        return [SearchHit(title=h["title"], url=h["url"], snippet=h["snippet"], source="core-rss", published_at=h.get("published_at", "")) for h in uniq[:total_limit]]

    def search_google_news(self, query: str, limit: int = 20) -> list[SearchHit]:
        q = (query or "").strip()
        if not q:
            return []

        # 영문(글로벌-US) 1회 우선 검색
        url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en&gl=US&ceid=US:en"
        try:
            raw = self._http_get(url)
        except Exception:
            _append_search_trace("google-news", q, request_url=url, region="global-us", status="fetch_error", hit_count=0)
            return []

        raw = html.unescape(raw)
        hits = self._extract_google_titles(raw, q, limit=limit * 2)
        _append_search_trace("google-news", q, request_url=url, region="global-us", status="ok", hit_count=len(hits))

        return [
            SearchHit(
                title=h["title"],
                url=h["url"],
                snippet=h["snippet"],
                source="google-news",
                published_at=h.get("published_at", ""),
            )
            for h in hits[:limit]
        ]

    def fetch(self, url: str) -> str:
        html_text = self._http_get(url)
        return self._strip_text(html_text)[:30000]


    def _decode_google_news_with_decoder(self, url: str, interval: int | None = None) -> str | None:
        if not _gnewsdecoder:
            return None
        if interval is None:
            try:
                interval = int(os.getenv("GOOGLE_NEWS_DECODER_INTERVAL", "1").strip())
            except Exception:
                interval = 1
        try:
            r = _gnewsdecoder(url, interval=interval)  # type: ignore[operator]
            if isinstance(r, dict) and r.get("status"):
                decoded = (r.get("decoded_url") or "").strip()
                if decoded.startswith(("http://", "https://")):
                    return decoded
        except Exception:
            return None
        return None

    def resolve_google_news_url(self, url: str) -> str | None:
        """Google News RSS 링크를 최종 원문 URL로 추적(브라우저 미사용 모드용)."""
        if not url:
            return None

        u = (url or "").strip()
        if "news.google.com/rss/articles" not in u and "news.google.com" not in u:
            return u

        # 1) 공개 라이브러리 기반 디코더 우선 시도
        decoded = self._decode_google_news_with_decoder(u)
        if decoded:
            return decoded

        # 2) google 래퍼는 단순 redirect만으로 원문이 안 떨어질 수 있어 본문 HTML에서 canonical/og/url을 더 탐색
        if requests is None:
            return u

        try:
            resp = requests.get(u, timeout=self.timeout_ms / 1000, headers=self._headers(), allow_redirects=True)
        except Exception:
            return u

        if resp is None:
            return u

        final = (resp.url or "").strip()
        if final and "news.google.com" not in final:
            return final

        text = resp.text or ""
        if not text:
            return final or u

        extracted = self._extract_google_news_target(text)
        if extracted:
            return extracted

        return final or u
