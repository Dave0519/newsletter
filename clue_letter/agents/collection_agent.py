from __future__ import annotations

import json
import re
import os
from collections import defaultdict
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Callable
from urllib.parse import urlparse
from pathlib import Path

from .models import CollectedArticle, UserProfile
from .utils import ensure_dir, safe_filename
from .llm_text_utils import summarize_ko, translate_ko, rewrite_title

SearchFn = Callable[[str, int], list]
FetchFn = Callable[[str], str]

NOISE_KEYWORDS = {
    "window.", "CQ_Analytics", "adobeDataLayer", "script", "function(", "style",
    "copyright", "cookie", "newsletter", "privacy", "personal", "navigation",
    "subscribe", "로그인", "로그아웃", "sign in", "sign up", "개인정보 처리방침"
}


def _is_noise(text: str) -> bool:
    if not text:
        return True
    t = (text or "").lower()
    if any(k.lower() in t for k in NOISE_KEYWORDS):
        return True
    compact = re.sub(r"\s+", "", text)
    if len(compact) > 120 and len(set(compact)) < 30:
        return True
    return False


def _is_candidate_news_url(url: str) -> bool:
    if not url:
        return False
    s = (url or "").strip().lower()
    if not s.startswith(("http://", "https://")):
        return False
    bad = {"#", "javascript:", "mailto:", "tel:"}
    if s in bad or s == "#" or s.startswith("javascript:"):
        return False
    try:
        p = urlparse(s)
    except Exception:
        return False
    path = (p.path or "").lower()
    if path in ("/", ""):
        return False
    # 카테고리/태그/검색 유입성 페이지 제거(필수 article 후보 우선)
    blocked_parts = ("/tag", "/category", "/categories", "/search", "/about", "/contact", "/login", "/signin")
    if path.endswith("/tag") or "/tag/" == path.rstrip("/"):
        return False
    for part in blocked_parts:
        if f"/{part.strip('/')}" == path.rstrip("/"):
            return False
    return True



def _to_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    v = (value or "").strip()
    if not v:
        return None
    try:
        dt = parsedate_to_datetime(v)
        if dt:
            return dt
    except Exception:
        pass
    for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
        try:
            d = datetime.strptime(v, fmt)
            if d.tzinfo is None:
                d = d.replace(tzinfo=datetime.now().astimezone().tzinfo)
            return d
        except Exception:
            pass
    try:
        d = datetime.fromisoformat(v.replace("Z", "+00:00"))
        return d
    except Exception:
        return None


def _is_within_last_days(value: str | None, max_days: int, now: datetime) -> bool:
    dt = _to_datetime(value)
    if dt is None:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=now.tzinfo)
    else:
        dt = dt.astimezone(now.tzinfo)
    return dt >= now - timedelta(days=max_days)


def _korean_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    if total == 0:
        return 0.0
    kor = len(re.findall(r"[가-힣]", text))
    return kor / float(total)


def _dedupe_title_in_body(title: str, body: str) -> str:
    """요약 단계에서 제목/헤더 중복이 summary로 들어가는 걸 최소화한다."""
    if not body:
        return ""

    text = re.sub(r"\s+", " ", (body or "")).strip()
    if not text:
        return ""

    # 1) 번역 본문 시작부에서 자주 나오는 메타/쿠키 블록 제거
    lower = text.lower()
    meta_marks = ["본 사이트의 쿠키 정보", "쿠키 정보", "개인정보", "privacy", "필수사항", "로그인", "로그아웃", "url 복사", "다음", "이전", "보러가기"]
    for mark in meta_marks:
        i = lower.find(mark)
        if 0 <= i <= 120:
            text = text[i + len(mark):].strip()
            lower = text.lower()

    # 2) 제목 기준 선두 제거(원문 제목과 닮은 prefix)
    t = (title or "").strip()
    t2 = re.split(r"\s[-|]\s", t, maxsplit=1)[0].strip()
    for cand in [t, t2, re.sub(r"\s+[|\-].*$", "", t2)]:
        cand = (cand or "").strip()
        if not cand:
            continue
        cl = cand.lower()
        if lower.startswith(cl):
            text = text[len(cand):].strip()
            lower = text.lower()

    # 3) 사이트명+타이틀이 함께 붙은 패턴("TITLE | SITE")
    if "|" in lower[:160]:
        parts = re.split(r"\s\|\s", text, maxsplit=1)
        if len(parts) > 1 and parts[0].strip().lower() in [p.lower() for p in [t, t2]]:
            text = parts[1].strip()
            lower = text.lower()

    # 4) 첫 구간이 완전히 같은 문구로 반복되는 케이스("ABAB ...")
    m = re.match(r"^(.{5,90}?)\s+\1\s+", text)
    if m:
        text = text[m.end():].strip()

    # 5) 과도한 반복문자 정리
    text = re.sub(r"([\w가-힣])\\1{10,}", r"\\1", text)
    return text[:18000]


def _trim_summary_prefix(title: str, summary: str) -> str:
    if not summary:
        return ""
    s = summary.strip()
    if not title:
        return s

    def _compact(t: str) -> str:
        return re.sub(r"[^가-힣a-zA-Z0-9]", "", t).lower()

    t1 = (title or "").strip()
    raw_parts = [
        t1,
        t1.strip().strip('"`').strip(),
        re.split(r"\s[-|]\s", t1, maxsplit=1)[0].strip() if t1 else "",
        re.sub(r"[|-].*$", "", t1).strip() if t1 else "",
    ]

    # title-like repeated prefix 제거
    for c in raw_parts:
        if not c:
            continue
        if s.startswith(c):
            s = s[len(c):].lstrip(" -|:/—–")
            break
        lc = c.lower()
        if s.lower().startswith(lc):
            s = s[len(lc):].lstrip(" -|:/—–")
            break
        if _compact(s).startswith(_compact(c)) and len(_compact(c)) >= 10:
            s = s[len(c):].lstrip(" -|:/—–")
            break

    # 앞부분 헤더 접두어 제거
    for p in ("Learn/", "기사보기 ", "기사입력", "다음 기사보기", "이전 기사보기"):
        if s.startswith(p):
            s = s[len(p):].strip()

    # 따옴표로 감싼 반복구간 제거(예: "title title")
    m = re.match(r"^\"([^\"]{6,})\"\s+", s)
    if m:
        quoted = m.group(1)
        tail = s[m.end():]
        ql = len(quoted)
        if tail.startswith(quoted):
            s = tail[ql:].strip()

    return s.strip()


class CollectionAgent:
    """사용자 맞춤 수집 엔진(주입된 수집/검색 어댑터에 따라 브라우저 모드/HTTP 모드로 동작)."""

    def __init__(
        self,
        data_root: Path,
        needs_agent,
        browser_search: SearchFn | None = None,
        fetch_body: FetchFn | None = None,
        resolve_url: Callable[[str], str | None] | None = None,
        search_core: SearchFn | None = None,
        search_google: SearchFn | None = None,
        min_count: int = 8,
        min_ratio: float = 0.008,
        max_days: int = 1,
        needs_target_per_keyword: int = 50,
        global_candidates_cap: int = 1200,
        source_fallback_weight: float = 0.82,
        min_success_body_len: int = 1200,
        daily_news_target: int = 10,
        daily_news_flexible_target: bool = True,
        global_fetch_time_cap: float = 90.0,
        source_fail_rate_circuit: float | None = 0.85,
    ):
        self.data_root = Path(data_root)
        self.needs_agent = needs_agent
        self.browser_search = browser_search
        self.fetch_body = fetch_body
        self.resolve_url = resolve_url
        self.search_core = search_core
        self.search_google = search_google
        self.min_count = min_count
        self.min_ratio = min_ratio
        self.max_days = max_days
        self.needs_target_per_keyword = needs_target_per_keyword
        self.global_candidates_cap = global_candidates_cap
        self.source_fallback_weight = source_fallback_weight
        self.min_success_body_len = min_success_body_len
        self.daily_news_target = daily_news_target
        self.daily_news_flexible_target = daily_news_flexible_target
        self.global_fetch_time_cap = float(global_fetch_time_cap) if global_fetch_time_cap is not None else 0.0
        self.source_fail_rate_circuit = float(source_fail_rate_circuit) if source_fail_rate_circuit is not None else None
        self.trace_enabled = os.getenv("CLUE_TRACE", "").lower() in {"1", "true", "yes", "on"}
        self.stage_counters = {"search_calls": 0, "fetch_calls": 0, "preselected": 0, "built": 0, "success_full": 0, "short": 0, "fail": 0, "filtered_stale": 0, "fallback_used": 0, "short_circuit": 0}

    def _log(self, msg: str) -> None:
        if self.trace_enabled:
            print(msg, flush=True)

    def _user_dirs(self, user: UserProfile) -> dict[str, Path]:
        base_name = safe_filename(user.name)
        base = self.data_root / base_name
        return {
            "base": base,
            "daily": base / "daily news",
            "history": base / "history",
        }

    def _to_kor(self, txt: str, max_chars: int = 2000) -> str:
        if not txt:
            return ""
        # translation fallback safe
        out = translate_ko(txt)
        return out[:max_chars] if out else txt[:max_chars]

    def _extract_country(self, text: str, title: str) -> str:
        t = f"{title} {text}".lower()
        country_map = {
            "미국": "US",
            "us ": "US",
            "united states": "US",
            "중국": "CN",
            "china": "CN",
            "대만": "TW",
            "taiwan": "TW",
            "한국": "KR",
            "korea": "KR",
            "일본": "JP",
            "japan": "JP",
            "유럽": "EU",
            "europe": "EU",
            "중동": "GLOBAL",
            "iran": "GLOBAL",
            "이스라엘": "GLOBAL",
            "uae": "GLOBAL",
            "사우디": "GLOBAL",
            "saudi": "GLOBAL",
            "global": "GLOBAL",
        }
        for k, v in country_map.items():
            if k in t:
                return v
        return "GLOBAL"

    def _split_sentences(self, text: str, max_sent: int = 5) -> str:
        cleaned = " ".join((text or "").replace("\n", " ").split())
        sent = re.split(r"(?<=[.!?])\s+", cleaned)
        return "\n".join([s.strip() for s in sent if s.strip()][:max_sent])

    def _build_quality_filter(self, title: str, summary: str, body: str) -> bool:
        if not title or not summary or not body:
            return False
        # 본문 길이: 너무 짧거나 너무 템플릿성인 문서는 버림
        if len(body) < 500:
            return False
        if _is_noise(body) or _is_noise(summary) or _is_noise(title):
            return False
        # 동일 반복 문자/토큰이 과도하게 반복되는 짧은 텍스트 제거
        if len(set(body.replace(" ", ""))) < 90:
            return False
        # 한글 비율(번역 결과가 영어로만 들어온 경우 제외)
        if _korean_ratio(body + summary + title) < self.min_ratio:
            return False
        return True

    def _extract_publication_time(self, raw: dict) -> str:
        for k in ("published_at", "pubDate", "date", "updated", "updated_at", "published"):
            v = raw.get(k) if isinstance(raw, dict) else None
            if isinstance(v, str):
                v = v.strip()
            elif v is not None:
                v = str(v).strip()
            if v:
                return v
        return ""

    def _compute_recency_score(self, published_at: str, now: datetime) -> float:
        dt = _to_datetime(published_at)
        if dt is None:
            return 0.0
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=now.tzinfo)
        else:
            dt = dt.astimezone(now.tzinfo)
        age_h = (now - dt).total_seconds() / 3600.0
        return max(0.0, 1.0 - (age_h / 24.0))

    def _score_preselect_by_metadata(self, title: str, snippet: str, needs: list[str], preferred_need: str | None = None) -> float:
        text = f"{title or ''} {snippet or ''}".lower()
        score = 0.0
        if preferred_need:
            p = preferred_need.lower().strip()
            if p and p in text:
                score += 1.2
        for idx, need in enumerate(needs):
            key = str(need).lower().strip()
            if not key:
                continue
            hits = text.count(key)
            if hits:
                score += 2.0 + idx * 0.05 + min(hits, 3) * 0.45
        return score

    def _score_relevance(self, article: CollectedArticle, interests: list[str], exclusions: list[str]) -> float:
        text = f"{article.title} {article.summary} {article.body}".lower()
        score = 0.0
        for idx, need in enumerate(interests):
            key = str(need).lower().strip()
            if not key:
                continue
            hits = text.count(key)
            if hits:
                score += 2.0 + idx * 0.05 + min(hits, 4) * 0.7
        for ex in exclusions:
            ex = str(ex).lower().strip()
            if ex and ex in text:
                score -= 100.0
        return score

    def _normalize_hit(self, hit):
        if isinstance(hit, dict):
            return {
                "title": (hit.get("title") or "").strip(),
                "url": (hit.get("url") or "").strip(),
                "snippet": (hit.get("snippet") or "").strip(),
                "published_at": (hit.get("published_at") or hit.get("pubDate") or "").strip(),
            }
        return {
            "title": (getattr(hit, "title", "") or "").strip(),
            "url": (getattr(hit, "url", "") or "").strip(),
            "snippet": (getattr(hit, "snippet", "") or "").strip(),
            "published_at": (getattr(hit, "published_at", "") or "").strip(),
        }

    def _build_time_prefixed_query(self, q: str, source_type: str, max_days: int) -> str:
        q = (q or "").strip()
        if not q or source_type != "google":
            return q

        # Google News 레벨에서 검색 단계부터 최근성 제한을 적용해 불필요 후보를 감소시킵니다.
        try:
            d = max(int(max_days), 1)
            if d <= 1:
                return f"{q} when:1d" if "when:" not in q.lower() else q
            if d <= 7:
                return f"{q} when:{d}d" if "when:" not in q.lower() else q
        except Exception:
            pass
        return q

    def _collect_need_hits(
        self,
        text: str,
        needs_payload: list[dict],
    ) -> tuple[list[str], list[str], list[str], float]:
        t = (text or "").lower()
        matched_need_ids: list[str] = []
        matched_needs: list[str] = []
        matched_aliases: list[str] = []
        need_score = 0.0
        for item in needs_payload:
            nid = str(item.get("need_id", "")).strip()
            aliases = [str(a).strip() for a in item.get("aliases", []) if str(a).strip()]
            for a in aliases:
                if a.lower() in t:
                    if nid and nid not in matched_need_ids:
                        matched_need_ids.append(nid)
                    if item.get("need_text") and item.get("need_text") not in matched_needs:
                        matched_needs.append(str(item.get("need_text")))
                    if a not in matched_aliases:
                        matched_aliases.append(a)
                    need_score += 1.0
                    break
        return matched_need_ids, matched_needs, matched_aliases, need_score

    def _maybe_build_article(
        self,
        raw: dict,
        source_query: str,
        needs_payload: list[dict],
        exclusions: list[str],
        collected_urls: set[str],
        source_type: str = "direct",
    ) -> CollectedArticle | None:
        url = str(raw.get("url") or "").strip()
        if not url or not _is_candidate_news_url(url) or url in collected_urls:
            return None
        try:
            p = urlparse(url)
            if p.scheme not in {"http", "https"}:
                return None
        except Exception:
            return None

        if "news.google.com/rss/articles/" in url and self.resolve_url:
            try:
                resolved = self.resolve_url(url)
            except Exception:
                resolved = None
            if resolved:
                if resolved in collected_urls or not _is_candidate_news_url(resolved):
                    return None
                url = resolved
                raw["url"] = resolved

        self.stage_counters["fetch_calls"] += 1
        self._log(f"[collect] fetch url={url[:90]} type={source_type}")
        try:
            body_text = self.fetch_body(url)
        except Exception as e:
            self.stage_counters["fail"] += 1
            self._log(f"[collect] fetch_failed url={url[:90]} err={type(e).__name__}")
            return None
        if not body_text:
            self.stage_counters["fail"] += 1
            self._log(f"[collect] fetch_empty url={url[:90]}")
            return None

        body_ko = self._to_kor(body_text, 20000)
        if not body_ko:
            self.stage_counters["fail"] += 1
            return None

        published_at = self._extract_publication_time(raw)
        body_len = len(body_ko)
        if body_len < int(self.min_success_body_len):
            self.stage_counters["short"] += 1
            return None

        raw_title_ko = self._to_kor((raw.get("title") or "").strip())
        if not raw_title_ko:
            self.stage_counters["fail"] += 1
            return None

        summary_body = _dedupe_title_in_body(raw_title_ko, body_ko)
        if not summary_body:
            summary_body = body_ko

        summary_ko = summarize_ko(summary_body, sentence_count=5)
        summary_ko = _trim_summary_prefix(raw_title_ko, summary_ko)
        title_ko = rewrite_title(raw_title_ko, body_ko + "\n" + summary_ko)
        if not title_ko:
            title_ko = raw_title_ko
        if not title_ko:
            title_ko = self._to_kor((raw.get("title") or "").strip())
        summary_ko = self._split_sentences(summary_ko, 5)

        if not self._build_quality_filter(title_ko, summary_ko, body_ko):
            self.stage_counters["fail"] += 1
            return None

        country = self._extract_country(body_ko, title_ko)
        merged = f"{title_ko} {summary_ko} {body_ko}"
        matched_need_ids, matched_needs, matched_aliases, need_hit_score = self._collect_need_hits(merged, needs_payload)

        article = CollectedArticle(
            title=title_ko,
            url=url,
            country=country,
            summary=summary_ko,
            body=body_ko,
            source="browser",
            source_type=source_type,
            need_category=matched_needs[0] if matched_needs else None,
            matched_need_ids=matched_need_ids,
            matched_needs=matched_needs,
            matched_aliases=matched_aliases,
            query=source_query,
            extracted_at=datetime.now().astimezone().isoformat(),
            published_at=published_at,
            body_len=body_len,
            extraction_status="success_full",
            recency_score=0.0,
            source_score=1.0 if source_type == "direct" else float(self.source_fallback_weight),
        )

        score = self._score_relevance(article, [n for n in (matched_needs or []) if n], exclusions)
        article.relevance_score = float(score + need_hit_score)
        article.need_match_score = article.relevance_score
        article.relevance_note = f"relevance_score={score:.1f}"
        if score <= 0:
            self.stage_counters["fail"] += 1
            return None

        if source_type == "google":
            article.source_score *= float(self.source_fallback_weight)

        collected_urls.add(url)
        self.stage_counters["success_full"] += 1
        return article


    def _score_total(self, article: CollectedArticle, now: datetime, needs_payload: list[dict], need_freq: dict[str, int], max_need_freq: int) -> float:
        recency = float(article.recency_score or 0.0)
        src_score = float(article.source_score or 1.0)
        need_score = float(len(article.matched_need_ids or []))

        # diversity penalty: 반복 니즈 과점유 완화
        local = 0
        if need_freq:
            if article.matched_need_ids:
                local = max(need_freq.get(nid, 0) for nid in article.matched_need_ids)
        penalty = 0.0
        if max_need_freq and local > 0:
            penalty = 0.15 * (local / float(max_need_freq))

        article.diversity_penalty = penalty
        return 0.55 * need_score + 0.25 * recency + 0.20 * src_score - penalty

    def _rebalance_by_need(self, articles: list[CollectedArticle], needs_payload: list[dict], target: int, flexible: bool = True) -> list[CollectedArticle]:
        if not articles:
            return []

        need_names = [it.get("need_id") for it in needs_payload if isinstance(it, dict) and it.get("need_id")]
        need_buckets: dict[str, list[CollectedArticle]] = {n: [] for n in need_names}
        unmatched = []

        for it in articles:
            used = False
            for nid in it.matched_need_ids:
                if nid in need_buckets:
                    need_buckets[nid].append(it)
                    used = True
            if not used:
                unmatched.append(it)

        for lst in need_buckets.values():
            lst.sort(key=lambda a: a.relevance_score, reverse=True)

        selected: list[CollectedArticle] = []
        used_urls = set()
        while True:
            progressed = False
            for nid, lst in need_buckets.items():
                while lst:
                    cand = lst.pop(0)
                    if cand.url in used_urls:
                        continue
                    selected.append(cand)
                    used_urls.add(cand.url)
                    progressed = True
                    break
                if len(selected) >= target:
                    break
            if len(selected) >= target or not progressed:
                break

        leftovers = [a for a in articles if a.url not in used_urls]
        leftovers.sort(key=lambda a: (a.relevance_score, a.body_len), reverse=True)
        while len(selected) < target and leftovers:
            cand = leftovers.pop(0)
            if cand.url in used_urls:
                continue
            selected.append(cand)
            used_urls.add(cand.url)

        if not flexible:
            return selected[:target]
        return selected

    def _build_total_news(
        self,
        candidates: list[CollectedArticle],
        needs_payload: list[dict],
    ) -> tuple[list[CollectedArticle], list[CollectedArticle]]:
        # 24h+니즈+요약 추출 success_full만 유지
        total_news = [a for a in candidates if a.extraction_status == "success_full"]

        now = datetime.now().astimezone()
        need_freq: dict[str, int] = {}
        for item in needs_payload:
            nid = str(item.get("need_id", "")).strip()
            if nid:
                need_freq[nid] = 0

        for a in total_news:
            a.recency_score = self._compute_recency_score(a.published_at, now)
            a.need_match_score = float(len(a.matched_need_ids))
            if a.matched_need_ids:
                for nid in a.matched_need_ids:
                    if nid in need_freq:
                        need_freq[nid] += 1
                a.source_score = max(0.1, float(a.source_score))

        max_need_freq = max(list(need_freq.values()) or [1])
        for a in total_news:
            a.relevance_score = self._score_total(a, now, needs_payload, need_freq, max_need_freq)

        total_news.sort(key=lambda x: x.relevance_score, reverse=True)
        selected = self._rebalance_by_need(total_news, needs_payload, target=int(self.daily_news_target), flexible=self.daily_news_flexible_target)

        # 저장용 total_news는 success_full만
        return selected, total_news

    def collect(self, user: UserProfile, limit: int = 25, min_count: int | None = None) -> list[CollectedArticle]:
        if self.fetch_body is None:
            raise RuntimeError("fetch_body가 주입되어야 수집을 시작할 수 있습니다.")
        if self.search_core is None and self.search_google is None:
            raise RuntimeError("search_core와 search_google 중 하나는 주입되어야 수집을 시작할 수 합니다.")

        min_count = int(min_count or self.min_count)
        latest_user = self.needs_agent.get_user(user.user_code)
        needs_payload = self.needs_agent.build_need_list_from_user(latest_user)
        needs_payload = [x for x in needs_payload if isinstance(x, dict) and x.get("aliases")]
        # 수집 채널별 니즈당 쿼리(직접/구글)
        query_pairs = self.needs_agent.build_need_queries(needs_payload, templates=["{}", "{} 뉴스", "{} 업데이트"])
        if not query_pairs:
            raise RuntimeError("유효한 needs가 없어 수집을 진행할 수 없습니다.")

        if len(query_pairs) > self.global_candidates_cap:
            query_pairs = query_pairs[: self.global_candidates_cap]

        candidates: list[CollectedArticle] = []
        collected_urls = set()
        now = datetime.now().astimezone()
        max_days = max(self.max_days, 1)

        # needs target per keyword
        needs_target = max(1, int(self.needs_target_per_keyword))
        per_need_cap = max(1, needs_target)

        self._log(f"[collect] start user_code={user.user_code} needs={len(needs_payload)} min_count={min_count} target={per_need_cap}")

        def stage_preselect(search_fn, pairs_in, source_type="direct", max_hits=18):
            self.stage_counters["search_calls"] += 1
            preselects = []
            for need_id, q in pairs_in:
                q_effective = self._build_time_prefixed_query(q, source_type, max_days)
                self._log(f"[collect] stage={source_type} need={need_id} query={q_effective}")
                try:
                    hits = search_fn(q_effective, limit=max_hits)
                except Exception as e:
                    self._log(f"[collect] stage={source_type} search_error={type(e).__name__} query={q_effective}")
                    hits = []

                per_need_added = 0
                seen_urls = set()
                for hit in hits[: int(self.global_candidates_cap / max(1, len(needs_payload)) + 4)]:
                    raw = self._normalize_hit(hit)
                    raw_url = (raw["url"] or "").strip()
                    if not raw_url or raw_url in collected_urls or raw_url in seen_urls:
                        continue
                    if not _is_candidate_news_url(raw_url):
                        continue
                    seen_urls.add(raw_url)
                    if not _is_within_last_days(raw.get("published_at") or "", max_days, now):
                        self.stage_counters["filtered_stale"] += 1
                        continue

                    score = self._score_preselect_by_metadata(raw.get("title") or "", raw.get("snippet") or "", [x.get("need_text", "") for x in needs_payload], preferred_need=need_id)
                    if score < self.min_ratio:
                        continue

                    preselects.append({
                        "score": score,
                        "need_id": need_id,
                        "query": f"{need_id}|{q}",
                        "raw": raw,
                        "source_type": source_type,
                    })
                    per_need_added += 1
                    self.stage_counters["preselected"] += 1
                    if per_need_added >= per_need_cap:
                        break

            return sorted(preselects, key=lambda x: x["score"], reverse=True)

        def should_stop(start_t: datetime, source_type: str, source_stats: dict[str, dict[str, int]]) -> bool:
            if self.global_fetch_time_cap > 0 and (datetime.now().astimezone() - start_t).total_seconds() >= self.global_fetch_time_cap:
                self.stage_counters["short_circuit"] += 1
                return True
            if self.source_fail_rate_circuit is not None and source_type in source_stats:
                s = source_stats[source_type]
                if s.get("attempt", 0) >= 8 and s.get("attempt", 0) > 0:
                    if (s["fail"] / s["attempt"]) >= self.source_fail_rate_circuit:
                        return True
            return False

        def build_from_preselect(preselects: list[dict], source_type: str, start_t: datetime, source_stats: dict[str, dict[str, int]]):
            for item in preselects:
                if len(candidates) >= min(self.global_candidates_cap, max(min_count, 12)):
                    break
                if should_stop(start_t, source_type, source_stats):
                    self._log(f"[collect] breaker triggered source={source_type}")
                    break

                source_stats.setdefault(source_type, {"attempt": 0, "fail": 0})
                source_stats[source_type]["attempt"] += 1

                raw = item["raw"]
                source_query = item["query"]
                src = item.get("source_type", source_type)
                article = self._maybe_build_article(raw, source_query, needs_payload, latest_user.exclusions, collected_urls, source_type=src)
                if article is None:
                    source_stats[source_type]["fail"] += 1
                    continue

                candidates.append(article)
                self.stage_counters["built"] += 1
                self._log(f"[collect] built score={article.relevance_score:.2f} need={article.need_category} src={src} url={article.url[:90]}")

        source_stats: dict[str, dict[str, int]] = {}
        start_t = datetime.now().astimezone()
        core_limit = limit

        # core first
        if callable(getattr(self, "search_core", None)):
            pre = stage_preselect(self.search_core, query_pairs, source_type="direct", max_hits=core_limit)
            build_from_preselect(pre, "direct", start_t, source_stats)

        # google fallback for deficient needs or if still low
        if callable(getattr(self, "search_google", None)):
            current = len(candidates)
            if current < max(min_count, 12):
                g_pairs = query_pairs[:]
                pre = stage_preselect(self.search_google, g_pairs, source_type="google", max_hits=max(8, limit // 2))
                self.stage_counters["fallback_used"] += 1
                build_from_preselect(pre, "google", start_t, source_stats)

        total_candidates = candidates
        selected, total_news = self._build_total_news(total_candidates, needs_payload)

        self._log(f"[collect] stage_summary preselected={self.stage_counters['preselected']} built={self.stage_counters['built']} success_full={self.stage_counters['success_full']} short={self.stage_counters['short']} fail={self.stage_counters['fail']} stale={self.stage_counters['filtered_stale']} fallback={self.stage_counters['fallback_used']} short_circuit={self.stage_counters['short_circuit']}")
        self._persist_total_news_and_daily(user, total_news, selected, total_candidates, needs_payload, min_count)

        final_selected = selected
        if len(final_selected) == 0:
            self._log(f"[collect] fail none selected")
            raise RuntimeError(f"수집 미달: 0건 (요청 24시간 대상 {min_count}건 미만)")

        # target is soft; 기본 목표 달성 못해도 가용 범위 내에서 반환
        target_cap = max(min_count, self.daily_news_target, 8)
        return final_selected[: target_cap]

    def _persist_total_news_and_daily(self, user: UserProfile, total_news: list[CollectedArticle], daily: list[CollectedArticle], total_candidates: list[CollectedArticle], needs_payload: list[dict], min_count: int) -> None:
        dirs = self._user_dirs(user)
        ensure_dir(dirs["daily"])
        ensure_dir(dirs["history"])

        issue = datetime.now().strftime("%Y-%m-%d")
        day_file = dirs["daily"] / f"{issue}.jsonl"

        with day_file.open("a", encoding="utf-8") as f:
            for a in daily:
                rec = self._make_daily_record(a)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        history_file = dirs["history"] / "titles.txt"
        with history_file.open("a", encoding="utf-8") as f:
            for a in daily:
                f.write(f"{a.title}\n")

        total_path = dirs["daily"] / f"total_news_{issue}.jsonl"
        with total_path.open("w", encoding="utf-8") as f:
            for a in total_news:
                rec = {
                    "url": a.url,
                    "title": a.title,
                    "published_at": a.published_at,
                    "source": a.source,
                    "source_type": a.source_type,
                    "matched_needs": a.matched_needs,
                    "matched_aliases": a.matched_aliases,
                    "need_match_score": a.need_match_score,
                    "source_score": a.source_score,
                    "diversity_penalty": a.diversity_penalty,
                    "body_len": a.body_len,
                    "collected_at": a.collected_at,
                    "query": a.query,
                    "extraction_status": a.extraction_status,
                    "recency_score": a.recency_score,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        metric_path = dirs["daily"] / "total_news_metrics.json"
        need_balance = {
            nid: 0 for nid in [x.get("need_id") for x in needs_payload if isinstance(x, dict) and x.get("need_id")]
        }
        for a in daily:
            for nid in a.matched_need_ids:
                need_balance[nid] = need_balance.get(nid, 0) + 1
        metrics = {
            "total_candidates": len(total_candidates),
            "need_matched_count": len([a for a in total_candidates if a.matched_need_ids]),
            "success_full_count": len(total_news),
            "success_full_rate": (len(total_news) / len(total_candidates)) if total_candidates else 0.0,
            "google_news_ratio": (len([a for a in total_candidates if (a.source_type or "").lower() == "google"]) / len(total_candidates)) if total_candidates else 0.0,
            "daily_news_need_balance": need_balance,
            "coverage_ok": (len(daily) >= min(min_count, self.daily_news_target)) if daily else False,
            "short_count": self.stage_counters.get("short", 0),
            "fail_count": self.stage_counters.get("fail", 0),
        }
        metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    def _persist_daily_and_history(self, user: UserProfile, articles: list[CollectedArticle]) -> None:
        dirs = self._user_dirs(user)
        ensure_dir(dirs["daily"])
        ensure_dir(dirs["history"])

        issue = datetime.now().strftime("%Y-%m-%d")
        day_file = dirs["daily"] / f"{issue}.jsonl"
        buckets = defaultdict(list)
        for a in articles:
            buckets[a.query or "default"].append(a)

        with day_file.open("a", encoding="utf-8") as f:
            for a in articles:
                rec = self._make_daily_record(a)
                rec["topics_coverage"] = sorted(buckets.keys())
                rec["coverage_count_for_topic"] = len(buckets[a.query or "default"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        history_file = dirs["history"] / "titles.txt"
        with history_file.open("a", encoding="utf-8") as f:
            for a in articles:
                f.write(f"{a.title}\n")

    def _make_daily_record(self, art: CollectedArticle) -> dict:
        # 핵심 공개 필드만 보관해 로그 데이터 노이즈를 줄임.
        # 원문 전체가 필요하면 처리 플로우에서 별도 로그/디버그로 처리하고,
        # 저장물은 제목/요약/링크 기반으로 최소화한다.
        summary_snip = (art.summary or "").replace("\n", " ").strip()
        if len(summary_snip) > 600:
            summary_snip = summary_snip[:597] + "..."

        return {
            "record_link": f"{art.url}|{art.title}",
            "title": art.title,
            "summary": art.summary,
            "summary_snippet": summary_snip,
            "url": art.url,
            "country": art.country,
            "query": art.query,
            "need_category": art.need_category,
            "matched_needs": art.matched_needs,
            "matched_aliases": art.matched_aliases,
            "need_keywords": sorted(set((art.matched_needs or []) + (art.matched_aliases or [])), key=lambda x: x),
            "extraction_status": art.extraction_status,
            "source_type": art.source_type,
            "collected_at": art.collected_at,
            "relevance_note": art.relevance_note,
            "source": art.source,
            "title_ko": art.title,
            "summary_ko": art.summary,
        }

    def load_daily_history(self, user: UserProfile, issue: str | None = None) -> list[dict]:
        issue = issue or datetime.now().strftime("%Y-%m-%d")
        path = self._user_dirs(user)["daily"] / f"{issue}.jsonl"
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out
