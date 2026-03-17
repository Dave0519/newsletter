from __future__ import annotations

import json
import re
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
    """브라우저 기반 사용자 맞춤 수집 엔진."""

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
        max_days: int = 2,
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

    def _maybe_build_article(self, raw: dict, source_query: str, interests: list[str], exclusions: list[str], collected_urls: set[str]) -> CollectedArticle | None:
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
                raw_title = raw.get("title") or ""
                raw["title"] = raw_title

        try:
            body_text = self.fetch_body(url)
        except Exception:
            return None
        if not body_text:
            return None

        body_ko = self._to_kor(body_text, 20000)
        raw_title_ko = self._to_kor((raw.get("title") or "").strip())
        if not raw_title_ko:
            return None

        # 요약 직전 본문 정제: title/헤더/메타성 반복 문구 제거
        summary_body = _dedupe_title_in_body(raw_title_ko, body_ko)
        if not summary_body:
            summary_body = body_ko

        summary_ko = summarize_ko(summary_body, sentence_count=5)
        summary_ko = _trim_summary_prefix(raw_title_ko, summary_ko)
        title_ko = rewrite_title(raw_title_ko, body_ko + "\n" + summary_ko)
        if not title_ko:
            # fallback to translated original title
            title_ko = raw_title_ko
        if not title_ko:
            # fallback to translated title
            title_ko = self._to_kor((raw.get("title") or "").strip())
        summary_ko = self._split_sentences(summary_ko, 5)

        if not self._build_quality_filter(title_ko, summary_ko, body_ko):
            return None

        country = self._extract_country(body_ko, title_ko)
        merged = f"{title_ko} {summary_ko} {body_ko}".lower()
        matched_need = None
        for need in interests:
            if need and need.lower() in merged:
                matched_need = need
                break
        if matched_need is None:
            matched_need = (source_query or "").split("|", 1)[0] or None

        article = CollectedArticle(
            title=title_ko,
            url=url,
            country=country,
            summary=summary_ko,
            body=body_ko,
            source="browser",
            need_category=matched_need,
            query=source_query,
        )

        score = self._score_relevance(article, interests, exclusions)
        article.relevance_score = float(score)
        article.relevance_note = f"relevance_score={score:.1f}"
        if score <= 0:
            return None

        collected_urls.add(url)
        return article

    def collect(self, user: UserProfile, limit: int = 25, min_count: int | None = None) -> list[CollectedArticle]:
        if self.fetch_body is None:
            raise RuntimeError("fetch_body가 주입되어야 수집을 시작할 수 있습니다.")
        if self.search_core is None and self.search_google is None:
            raise RuntimeError("search_core와 search_google 중 하나는 주입되어야 수집을 시작할 수 있습니다.")

        min_count = int(min_count or self.min_count)
        # NeedsAgent와 직접 동기화해서 최신 니즈 상태 반영
        latest_user = self.needs_agent.get_user(user.user_code)
        interests = [x.strip() for x in (latest_user.interests or []) if x and x.strip()]
        # NeedsAgent 기준 최신 인터레스트로 동기화하고, 1차 코어는 니즈당 1개 쿼리만 사용
        query_pairs = self.needs_agent.build_need_queries(interests, templates=["{}"])
        if not query_pairs:
            raise RuntimeError("유효한 needs가 없어 수집을 진행할 수 없습니다.")

        candidates: list[CollectedArticle] = []
        bucket_by_need: dict[str, list[CollectedArticle]] = {need: [] for need in interests}
        now = datetime.now().astimezone()
        max_days = max(self.max_days, 1)
        collected_urls = set()

        candidate_target = max(min_count, 8)
        per_need_prefetch = max(3, (candidate_target // max(len(interests), 1)) + 1)
        preselect_limit = per_need_prefetch * max(len(interests), 1)

        def stage_preselect(search_fn, query_pairs_in, max_hits: int = 25):
            preselects: list[dict] = []
            for need, q in query_pairs_in:
                try:
                    hits = search_fn(q, limit=max_hits)  # type: ignore[attr-defined]
                except Exception:
                    hits = []

                for hit in hits:
                    raw = self._normalize_hit(hit)
                    if not raw["url"] or not _is_candidate_news_url(raw["url"]) or raw["url"] in collected_urls:
                        continue
                    published_at = raw.get("published_at") or ""
                    if not _is_within_last_days(published_at, max_days, now):
                        continue

                    raw_score = self._score_preselect_by_metadata(
                        raw.get("title") or "",
                        raw.get("snippet") or "",
                        interests,
                        preferred_need=need,
                    )
                    # 제목 검색어 필터링이 너무 강해지지 않도록 fallback를 보장
                    if raw_score <= 0 and len(preselects) >= preselect_limit * 2:
                        continue

                    preselects.append({
                        "score": raw_score,
                        "query": f"{need}|{q}",
                        "need": need,
                        "raw": raw,
                    })

            return sorted(preselects, key=lambda x: x["score"], reverse=True)[:preselect_limit]

        def build_from_preselect(preselects: list[dict]) -> None:
            for item in preselects:
                if len(candidates) >= max(min_count, 12):
                    break
                raw = item["raw"]
                source_query = item["query"]
                article = self._maybe_build_article(raw, source_query, interests, latest_user.exclusions, collected_urls)
                if article is None:
                    continue
                matched_need = article.need_category
                candidates.append(article)
                if matched_need in bucket_by_need:
                    bucket_by_need[matched_need].append(article)

        # 1차: CORE 채널
        if hasattr(self, "search_core") and callable(getattr(self, "search_core", None)):
            pre = stage_preselect(self.search_core, query_pairs, max_hits=limit)
            build_from_preselect(pre)

        # 2차: 부족한 니즈만 Google News 보강
        if hasattr(self, "search_google") and callable(getattr(self, "search_google", None)):
            need_min = max(1, (min_count + max(len(interests), 1) - 1) // max(len(interests), 1))
            deficient_needs = [need for need in interests if len(bucket_by_need.get(need, [])) < need_min]

            # 보강은 부족한 니즈만 진행
            if deficient_needs:
                deficit_pairs = [(need, need) for need in deficient_needs]
                pre = stage_preselect(self.search_google, deficit_pairs, max_hits=limit)
                build_from_preselect(pre)

        ordered = sorted(candidates, key=lambda x: x.relevance_score, reverse=True)
        selected = ordered[: max(min_count, 12)]
        if len(selected) < min_count:
            raise RuntimeError(f"수집 미달: {len(selected)}건 (요청 24시간 대상 {min_count}건 미만)")

        self._persist_daily_and_history(user, selected)
        return selected[: max(min_count, 8)]

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
            "record_link": f"{art.title}:{art.summary}:{art.url}:{art.country}",
            "title": art.title,
            "summary": art.summary,
            "summary_snippet": summary_snip,
            "url": art.url,
            "country": art.country,
            "query": art.query,
            "need_category": art.need_category,
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
