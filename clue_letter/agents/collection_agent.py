from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
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
    "subscribe", "로그인", "로그아웃", "sign in", "sign up"
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


def _korean_ratio(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    if total == 0:
        return 0.0
    kor = len(re.findall(r"[가-힣]", text))
    return kor / float(total)


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

    def collect(self, user: UserProfile, limit: int = 25, min_count: int | None = None) -> list[CollectedArticle]:
        if self.browser_search is None or self.fetch_body is None:
            raise RuntimeError("browser_search와 fetch_body가 모두 주입되어야 수집을 시작할 수 있습니다.")

        min_count = int(min_count or self.min_count)
        interests = [x.strip() for x in (user.interests or []) if x and x.strip()]
        base_queries = self.needs_agent.ensure_queries_by_interests(interests)

        # 기본 니즈별 쿼리 맵 재구성(공급부족 니즈 추적용)
        need_templates = ["{}", "{} news", "{} AI", "{} 업데이트", "{} 현황"]
        need_query_pairs: list[tuple[str, str]] = []
        for need in interests:
            for t in need_templates:
                need_query_pairs.append((need, t.format(need)))
        # 중복 제거
        deduped: list[tuple[str, str]] = []
        seen_queries = set()
        for need, q in need_query_pairs:
            q2 = q.strip()
            if not q2 or q2 in seen_queries:
                continue
            seen_queries.add(q2)
            deduped.append((need, q2))
        query_pairs = deduped

        candidates: list[CollectedArticle] = []
        bucket_by_need: dict[str, list[CollectedArticle]] = {need: [] for need in interests}
        collected_urls = set()

        # 1차 기준: 코어 소스 우선 (활성 코어 피드)
        def add_from_query(hits, stage_name: str, source_query: str):
            for hit in hits:
                if len(candidates) >= max(min_count, 12):
                    return

                raw = hit if isinstance(hit, dict) else {
                    "title": getattr(hit, "title", ""),
                    "url": getattr(hit, "url", ""),
                    "snippet": getattr(hit, "snippet", ""),
                }
                url = str(raw.get("url") or "").strip()
                if not url or url in collected_urls:
                    continue
                p = urlparse(url)
                if p.scheme not in {"http", "https"}:
                    continue

                # Google News RSS 링크면 원문 링크로 resolve
                if "news.google.com/rss/articles/" in url and self.resolve_url:
                    try:
                        resolved = self.resolve_url(url)
                    except Exception:
                        resolved = None
                    if resolved:
                        if resolved in collected_urls:
                            continue
                        url = resolved
                        p = urlparse(url)
                        if p.scheme not in {"http", "https"}:
                            continue
                        raw["url"] = resolved

                try:
                    body_text = self.fetch_body(url)
                except Exception:
                    continue
                if not body_text:
                    continue

                body_ko = self._to_kor(body_text, 20000)
                title_ko = self._to_kor((raw.get("title") or "").strip())
                if not title_ko:
                    continue
                summary_ko = summarize_ko(body_ko, sentence_count=5)
                title_ko = rewrite_title(title_ko, body_ko + "\n" + summary_ko)
                if not title_ko:
                    # fallback to translated title
                    title_ko = self._to_kor((raw.get("title") or "").strip())
                summary_ko = self._split_sentences(summary_ko, 5)

                if not self._build_quality_filter(title_ko, summary_ko, body_ko):
                    continue

                country = self._extract_country(body_ko, title_ko)
                matched_need = None
                merged = f"{title_ko} {summary_ko} {body_ko}".lower()
                for need in interests:
                    if need and need.lower() in merged:
                        matched_need = need
                        break
                if matched_need is None:
                    query_need = source_query.split('|', 1)[0]
                    matched_need = query_need

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

                score = self._score_relevance(article, interests, user.exclusions)
                article.relevance_score = float(score)
                article.relevance_note = f"relevance_score={score:.1f}"
                if score <= 0:
                    continue

                collected_urls.add(url)
                candidates.append(article)
                if matched_need in bucket_by_need:
                    bucket_by_need[matched_need].append(article)

        # 1차: CORE 채널
        if hasattr(self, "search_core") and callable(getattr(self, "search_core", None)):
            for need, q in query_pairs:
                if len(candidates) >= max(min_count, 12):
                    break
                try:
                    hits = self.search_core(q, limit=limit)  # type: ignore[attr-defined]
                except Exception:
                    hits = []
                add_from_query(hits, "core", f"{need}|{q}")

        # 1차 실패시 기존 검색으로 보완
        if len(candidates) < max(min_count, 12):
            fallback_queries = base_queries[:limit]
            for q in fallback_queries:
                if len(candidates) >= max(min_count, 12):
                    break
                hits = self.browser_search(q, limit=limit)
                add_from_query(hits, "fallback", q)

        # 2차: 부족한 니즈만 Google News 보강
        if hasattr(self, "search_google") and callable(getattr(self, "search_google", None)):
            need_min = max(1, (min_count + max(len(interests), 1) - 1) // max(len(interests), 1))
            for need in interests:
                if len(candidates) >= max(min_count, 12):
                    break
                if len(bucket_by_need.get(need, [])) >= need_min:
                    continue
                need_query = need
                try:
                    hits = self.search_google(need_query, limit=limit)  # type: ignore[attr-defined]
                except Exception:
                    hits = []
                add_from_query(hits, "google-news", need_query)

        # 24h 미만로 제한되더라도 최신성 판단은 저장시각 컬럼으로 정렬
        now = datetime.now()
        cutoff = now - timedelta(hours=24 * max(self.max_days, 1))
        recent = [c for c in candidates if c.collected_at >= cutoff.isoformat()]

        ordered = sorted(candidates, key=lambda x: x.relevance_score, reverse=True)
        selected = recent if len(recent) >= min_count else ordered

        selected = selected[: max(min_count, 12)]
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
