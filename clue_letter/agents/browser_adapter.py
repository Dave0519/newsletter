from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from urllib.parse import quote_plus, unquote, urlparse, parse_qs
from typing import Any, Optional


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    source: str = "browser"


def _run_browser_cmd(*args: str, timeout_ms: int = 40000) -> str:
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


class BrowserRelayAdapter:
    def __init__(self, search_engine: str = "duckduckgo", open_timeout_ms: int = 20000):
        self.search_engine = search_engine
        self.open_timeout_ms = open_timeout_ms
        self._last_target: Optional[str] = None
        self._started = False
        self._ensure_start()

    def _ensure_start(self):
        if self._started:
            return
        _run_browser_cmd("start")
        self._started = True

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
        out = _run_browser_cmd("open", url, "--json", timeout_ms=self.open_timeout_ms)
        data = _json_load_last(out)
        if isinstance(data, dict):
            tid = data.get("targetId")
            if tid:
                self._last_target = tid
                self._wait_load(tid, "domcontentloaded", timeout_ms=8000)
        return out

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



def _parse_feed_datetime(raw: str):
    raw = (raw or '').strip()
    if not raw:
        return None
    s = raw.strip()
    for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"]:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    # XML feed may include microseconds or no tz
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

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
        out: list[dict[str, str]] = []
        query_low = (query or "").lower()

        # RSS item parser (XML)
        items = re.findall(r"<item>(.*?)</item>", text, flags=re.S | re.I)
        if items:
            for item in items:
                title = re.search(r"<title[^>]*>(.*?)</title>", item, flags=re.S | re.I)
                link = re.search(r"<link[^>]*>(.*?)</link>", item, flags=re.S | re.I)
                desc = re.search(r"<description[^>]*>(.*?)</description>", item, flags=re.S | re.I)
                pub = re.search(r"<pubDate[^>]*>(.*?)</pubDate>", item, flags=re.S | re.I)
                dt = re.search(r"<dc:date[^>]*>(.*?)</dc:date>", item, flags=re.S | re.I)
                if not title or not link:
                    continue
                t = re.sub(r"\s+", " ", re.sub(r"<!\[CDATA\[(.*?)\]>", r"\1", title.group(1))).strip()
                l = re.sub(r"\s+", " ", re.sub(r"<!\[CDATA\[(.*?)\]>", r"\1", link.group(1))).strip()
                d = re.sub(r"\s+", " ", re.sub(r"<!\[CDATA\[(.*?)\]>", r"\1", (desc.group(1) if desc else ""))).strip()
                p = re.sub(r"\s+", " ", re.sub(r"<!\[CDATA\[(.*?)\]>", r"\1", (pub.group(1) if pub else (dt.group(1) if dt else "")))).strip()
                blob = f"{t} {d}".lower()
                if query and query_low not in blob and all(k not in blob for k in query_low.split()):
                    continue
                if l:
                    out.append({"title": t, "url": l, "snippet": d, "source": "rss", "published_at": p})

        # fallback if xml parse fails
        if not out:
            for link, txt in re.findall(r"<a[^>]+href=['\"\']([^'\"]+)['\"\'][^>]*>(.*?)</a>", text, flags=re.S | re.I):
                t = re.sub(r"\s+", " ", re.sub(r"<!\[CDATA\[(.*?)\]>", r"\1", txt)).strip()
                l = re.sub(r"\s+", " ", re.sub(r"<!\[CDATA\[(.*?)\]>", r"\1", link)).strip()
                if not t or not l:
                    continue
                if query and query_low not in t.lower() and query_low not in l.lower():
                    continue
                out.append({"title": t, "url": l, "snippet": "", "source": "rss"})

        uniq: list[dict[str, str]] = []
        seen = set()
        for item in out[: limit * 4]:
            u = item.get("url")
            if not u or not isinstance(u, str) or u in seen:
                continue
            seen.add(u)
            uniq.append(item)
            if len(uniq) >= limit:
                break
        return uniq



    def search_google_news(self, query: str, limit: int = 20) -> list[SearchHit]:
        q = (query or "").strip()
        if not q:
            return []
        url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=ko&gl=KR&ceid=KR:ko"
        try:
            self._open(url)
        except Exception:
            return []

        try:
            data = self._run_with_target('() => ({text:(document.documentElement? document.documentElement.innerHTML:"" )})', context="gnews", max_retry=1, eval_timeout_ms=12000)
        except Exception:
            return []

        raw = data.get("result", "") if isinstance(data, dict) else ""
        if not isinstance(raw, str):
            return []

        hits = self._extract_google_titles(raw, q, limit=limit*2)
        return [SearchHit(title=h["title"], url=h["url"], snippet=h["snippet"], source="google-news") for h in hits][:limit]

    def search_core_rss(self, query: str, feed_urls: list[str], limit: int = 20) -> list[SearchHit]:
        q = (query or "").strip()
        if not q or not feed_urls:
            return []

        all_hits: list[dict[str, str]] = []
        for feed_url in feed_urls[:20]:
            try:
                self._open(feed_url)
            except Exception:
                continue
            try:
                data = self._run_with_target('() => ({text:(document.documentElement? document.documentElement.innerHTML:"" )})', context="core-rss", max_retry=1, eval_timeout_ms=12000)
            except Exception:
                continue
            raw = data.get("result", "") if isinstance(data, dict) else ""
            if not isinstance(raw, str):
                continue
            all_hits.extend(self._extract_google_titles(raw, q, limit=limit*3))

        uniq: list[dict[str, str]] = []
        seen=set()
        for h in all_hits:
            url=h.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            uniq.append(h)
            if len(uniq)>=limit:
                break
        return [SearchHit(title=h["title"], url=h["url"], snippet=h["snippet"], source="core-rss") for h in uniq]


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
