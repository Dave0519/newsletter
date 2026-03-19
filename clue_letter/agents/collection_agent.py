from __future__ import annotations

import json
import re
import os
from collections import defaultdict
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Callable
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode, unquote_plus
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from pathlib import Path
import ssl
import hashlib
import requests

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

PROMOTION_NOISE_KEYWORDS = {
    "쿠폰", "할인", "세일", "특가", "경품", "이벤트", "프로모션", "기획전", "특별가", "개통", "혜택",
    "출시 행사", "행사", "gift", "discount", "sale", "coupon", "voucher", "giveaway", "preorder",
    "pre-order", "limited time", "free shipping", "special offer", "exclusive deal", "가격", "판매", "구매", "구매하기",
    "신제품", "신규가입", "회원혜택", "회원 특전", "런칭", "출시", "이벤트 페이지", "신청", "예약 구매", "예약판매"
}

PROMO_URL_HINTS = {
    "/event", "/promo", "/promotion", "/offers", "/offer", "/deal", "/shop", "/shopping", "/store", "/product", "/buy", "/coupon"
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


def _is_marketing_content(text: str, url: str | None = None) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    if any(k in t for k in PROMOTION_NOISE_KEYWORDS):
        return True

    # 가격/할인 패턴 (금액+할인/세일 조합)
    if re.search(r"\d+(\s?)(%|퍼센트)(\s*할인|\s*세일|\s*할인률)?", t):
        return True
    if re.search(r"(\$|\u20a9|원)\s*\d+[\d,\.]*(\s*[~-]?\s*(\$|\u20a9|원)\s*\d+)?", t):
        return True

    if url:
        u = (url or "").lower()
        if any(u.find(hint) != -1 for hint in PROMO_URL_HINTS):
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
    m = re.match(r'^"([^"]{6,})"\s+', s)
    if m:
        quoted = m.group(1)
        tail = s[m.end():]
        ql = len(quoted)
        if tail.startswith(quoted):
            s = tail[ql:].strip()

    return s.strip()


def _normalize_title_signature(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    t = re.sub(r"[^가-힣a-z0-9]+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:90]



def _make_article_key(title: str, url: str) -> str:
    sig = _normalize_title_signature(title)
    if sig:
        sig_hash = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:10]
    else:
        sig_hash = ""
    u = (url or "").strip().lower()
    return f"{u}::{sig_hash}"


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
        total_news_target: int = 50,
        source_fallback_weight: float = 0.82,
        min_success_body_len: int = 1200,
        daily_news_target: int = 10,
        daily_news_flexible_target: bool = True,
        query_batch_size: int = 8,
        global_fetch_time_cap: float = 45.0,
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
        self.total_news_target = max(1, int(total_news_target))
        self.source_fallback_weight = source_fallback_weight
        self.min_success_body_len = min_success_body_len
        self.daily_news_target = daily_news_target
        self.daily_news_flexible_target = daily_news_flexible_target
        self.query_batch_size = max(0, int(query_batch_size))
        self.global_fetch_time_cap = float(global_fetch_time_cap) if global_fetch_time_cap is not None else 0.0
        self.source_fail_rate_circuit = float(source_fail_rate_circuit) if source_fail_rate_circuit is not None else None
        self.trace_enabled = os.getenv("CLUE_TRACE", "").lower() in {"1", "true", "yes", "on"}
        self.stage_counters = {"search_calls": 0, "fetch_calls": 0, "preselected": 0, "built": 0, "success_full": 0, "short": 0, "fail": 0, "filtered_stale": 0, "filtered_unreachable": 0, "fallback_used": 0, "short_circuit": 0, "normalized_total": 0}

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

    def _llm_request(self, prompt: str, max_tokens: int = 240, temperature: float = 0.0) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return ""
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if str(model).lower().startswith("gpt-5"):
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            if r.status_code != 200:
                return ""
            return ((r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip())
        except Exception:
            return ""

    def load_daily_news(self, user: UserProfile, issue: str | None = None) -> list[CollectedArticle]:
        """오늘/특정 일자의 daily_news를 파일로부터 다시 로드한다."""
        dirs = self._user_dirs(user)
        if issue is None:
            issue = datetime.now().strftime("%Y-%m-%d")
        path = dirs["daily"] / f"{issue}.jsonl"
        if not path.exists():
            return []

        items: list[CollectedArticle] = []
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            record_link = str(row.get("record_link", "")).strip()
            # record_link may contain pre-composed "url|title"
            url = record_link.split("|", 1)[0].strip()
            matched_needs = row.get("matched_needs")
            if not isinstance(matched_needs, list):
                matched_needs = []
            matched_need_ids = row.get("matched_need_ids")
            if not isinstance(matched_need_ids, list):
                matched_need_ids = []

            items.append(
                CollectedArticle(
                    title=str(row.get("title") or ""),
                    url=url,
                    country=str(row.get("country") or "GLOBAL"),
                    summary=str(row.get("summary") or ""),
                    body="",
                    source=str(row.get("source") or "http"),
                    source_type=str(row.get("source_type") or "direct"),
                    need_category=row.get("need_category") or row.get("need") or None,
                    matched_need_ids=matched_need_ids,
                    matched_needs=matched_needs,
                    matched_aliases=row.get("matched_aliases") if isinstance(row.get("matched_aliases"), list) else [],
                    query=str(row.get("query") or ""),
                    published_at=str(row.get("published_at") or ""),
                    extraction_status=str(row.get("extraction_status") or "success_full"),
                    relevance_score=float(row.get("need_match_score") or 0.0),
                    need_match_score=float(row.get("need_match_score") or 0.0),
                    source_score=float(row.get("source_score") or 1.0),
                    relevance_note=str(row.get("relevance_note") or ""),
                    collected_at=str(row.get("collected_at") or datetime.now().isoformat()),
                    recency_score=float(row.get("recency_score") or 0.0),
                    body_len=int(row.get("body_len") or 0),
                )
            )
        return items


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
        # summary는 번역 산출물 특성상 일시적으로 단정/고정 텍스트가 들어올 수 있어
        # 본문+제목 품질만 강하게 걸러내고 summary는 보조 지표로 처리
        if _is_noise(body) or _is_noise(title):
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

    def _is_url_accessible(self, url: str, timeout: float = 8.0, read_bytes: int = 2048) -> bool:
        if not _is_candidate_news_url(url):
            return False

        headers = {"User-Agent": "Mozilla/5.0"}
        for method in ("HEAD", "GET"):
            try:
                req = Request(url, method=method, headers=headers)
                with urlopen(req, timeout=timeout, context=ssl._create_unverified_context()) as response:
                    status = getattr(response, "status", 200)
                    if status >= 400:
                        continue
                    if method == "GET":
                        if not response.read(read_bytes):
                            continue
                    return True
            except HTTPError as e:
                if method == "HEAD" and e.code in {403, 405}:
                    continue
                if method == "HEAD" and 400 <= e.code < 500:
                    continue
            except URLError:
                continue
            except Exception:
                return False
        return False

    def _is_resolved_news_url(self, url: str) -> bool:
        u = (url or "").strip().lower()
        if not u:
            return False
        # Google News 래퍼 잔존 링크는 아직 원문으로 보지 않음
        if "news.google.com/rss/articles" in u or "news.google.com/articles" in u:
            return False
        return _is_candidate_news_url(url)

    def _canonicalize_url(self, url: str) -> str:
        s = (url or "").strip()
        if not s:
            return ""
        try:
            p = urlparse(s)
            if not p.scheme or not p.netloc:
                return s

            netloc = (p.netloc or "").strip().lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]

            path = (p.path or "").strip()
            # 모바일/AMP/트래커 경로 패턴 정리
            path = re.sub(r"/amp/?$", "/", path, flags=re.I)
            path = re.sub(r"/m/?$", "/", path, flags=re.I)
            path = re.sub(r"/index\.html$", "/", path, flags=re.I)
            path = re.sub(r"/+", "/", path or "/")

            # 추적 파라미터 제거 (강화)
            drop_prefixes = ("utm_", "utm", "fbclid", "gclid", "ref", "oc", "ref_src", "amp", "__cf_chl_tk", "session", "sid", "igsh", "mc_cid", "mc_eid", "sp", "sp_a", "sp_k", "sp_r")
            q = parse_qs(p.query, keep_blank_values=False)
            cleaned = []
            for key, values in q.items():
                lk = key.lower()
                if lk in drop_prefixes or any(lk.startswith(pref) for pref in drop_prefixes):
                    continue
                for v in values:
                    # 빈 값이더라도 구조 동일성 유지하려면 유지
                    if v is None:
                        v = ""
                    cleaned.append((key, v))

            # google news wrapper는 파라미터 제거 후 url 자체를 더 이상 붙잡지 않는다
            if "news.google.com" in p.netloc.lower() and not p.path.lower().startswith("/url"):
                path = p.path

            return urlunparse((p.scheme.lower(), netloc, path, p.params, urlencode(cleaned, doseq=True), "")).rstrip("?")
        except Exception:
            return s

    def _normalize_article_url(self, url: str) -> str:
        resolved = (url or "").strip()
        if not resolved:
            return ""

        # Google 래퍼 링크/유사 링크는 원문 URL로 해석 시도
        if "news.google.com/rss/articles" in resolved.lower() or "news.google.com" in resolved.lower():
            try:
                parsed = urlparse(resolved)
                q = parse_qs(parsed.query)
                candidates = []
                for key in ("url", "u", "rurl", "ru", "aurl"):
                    for v in q.get(key, []):
                        if v:
                            candidates.append(v)
                # google 내부 추적 파라미터가 있을 때만 추출
                for v in candidates:
                    cand = unquote_plus(v).strip()
                    if cand:
                        resolved = cand
                        break
            except Exception:
                pass

        if self.resolve_url:
            try:
                out = self.resolve_url(resolved)
                if out:
                    resolved = out
            except Exception:
                pass

        resolved = self._canonicalize_url(resolved)
        return resolved

    def _normalize_topic_key(self, topic: str) -> str:
        t = (topic or "").strip()
        if not t:
            return ""
        # (1) 문자 정리
        t = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip().lower()

        # (2) 너무 다양한 표현을 줄이기 위한 stopword 제거 + 앞 토큰 위주로 키를 거칠게 만들기
        stop = {
            "관련", "동향", "이슈", "업데이트", "뉴스", "브리핑", "정리", "요약",
            "발표", "추진", "검토", "확대", "강화", "협력", "계획", "전망",
            "report", "news", "update", "briefing"
        }
        tokens = [x for x in t.split(" ") if x and x not in stop]
        # 핵심 토큰 일부만 사용(키를 coarse하게 만들어 같은 사건이 같은 버킷으로 들어오게 함)
        tokens = tokens[:6]
        key = " ".join(tokens).strip()
        return key[:60]

    def _llm_is_same_topic(self, left: CollectedArticle, right: CollectedArticle) -> bool:
        l_body = (left.body or "").replace("\n", " ").strip()[:500]
        r_body = (right.body or "").replace("\n", " ").strip()[:500]
        if not l_body:
            l_body = (left.summary or "").replace("\n", " ").strip()[:500]
        if not r_body:
            r_body = (right.summary or "").replace("\n", " ").strip()[:500]

        prompt = (
            "역할 : 뉴스레터 에디터\n"
            "작업 : 아래 두 기사의 주제가 동일 기사군(동일 사/건, 동일 사건·동일 핵심 인물/기술/기관)인지 판단하세요.\n"
            "정답은 'SAME' 또는 'DIFFERENT' 중 하나만 출력하세요.\n"
            "동일 주제 기준: 핵심 주제, 핵심 사건, 핵심 대상이 본질적으로 동일하면 SAME.\n\n"
            f"기사A\n제목: {left.title or ''}\n요약: {left.summary or ''}\n본문: {l_body}\n\n"
            f"기사B\n제목: {right.title or ''}\n요약: {right.summary or ''}\n본문: {r_body}"
        )
        out = self._llm_request(prompt).strip().lower()
        if not out:
            return False
        if out.startswith("same") or out.startswith("duplicate"):
            return True
        if "same" in out.split() and "different" not in out:
            return True
        return False

    def _llm_judge_topic(self, title: str, summary: str, body: str = "") -> str:
        body_snip = (body or "").replace("\\n", " ").strip()[:500]
        if not body_snip:
            body_snip = (summary or "").replace("\\n", " ").strip()[:500]
        base = (
            f"제목: {title or ''}\\n"
            f"요약: {summary or ''}\\n"
            f"본문(요약/추출본/short): {body_snip}"
        ).strip()
        if not base:
            return ""

        prompt = (
            "역할 : 뉴스레터 에디터\\n"
            "작업 : 아래 기사 제목 목록에서 동일 사/건인/물발표를 다루는 중복 기사를 필터링하세요.\\n"
            "중복 판단 기준 :\\n"
            "제목에서 추출한 3가지 요소(핵심 주제, 핵심 사건, 핵심 대상) 중 1개 이상이 일치하면 중복으로 처리합니다.\\n"
            "특히 제목/요약/본문(요약/추출본/short)에 공통 핵심 키워드가 3개 이상(인물명/기술명/행사명) 겹치면 동일 주제 중복으로 판단하세요.\n"
            "동일 인물/기관/기술과 동일 사건이 함께 2개 이상 결합되면 무조건 중복으로 처리하세요.\n\n"
            "핵심 주제(인/물기업)\\n"
            "핵심 사건(방문/발표/협력 )등\\n"
            "핵심 대상(제/품기술/장소)\\n\\n"
            "정치 제외 판단 기준 :\\n"
            "핵심 대상(제/품기술/장소)과 상관없이 제목에 정치 관련 키워드(예:선거,출마,공약,국회,대사관,전쟁범죄 등)가 포함되면 전체 제외로 판단하세요.\\n"
            "정치 기사면 문자열 '정치기사제외'만 출력하고, 그 외 중복 판단 주제는 한 줄로만 출력하세요.\\n"
            f"{base}"
        )
        out = self._llm_request(prompt).strip()
        if not out:
            return ""
        out = re.sub(r"\s+", " ", out).strip()
        # 한 줄 + 첫 토막만 사용
        out = out.replace("\n", " ").strip()
        if len(out) > 40:
            out = out[:40]
        return out
    def _dedupe_daily_by_topic(self, articles: list[CollectedArticle]) -> list[CollectedArticle]:
        if not articles:
            return []

        def _token_set(a: CollectedArticle) -> set[str]:
            base = f"{a.title or ''} {a.summary or ''}".lower()
            # 2글자 이상 토큰만
            return set(re.findall(r"[가-힣a-z0-9]{2,}", base))

        def _jaccard(sa: set[str], sb: set[str]) -> float:
            if not sa or not sb:
                return 0.0
            inter = len(sa & sb)
            union = len(sa | sb)
            return inter / float(union) if union else 0.0

        cache: dict[str, str] = {}
        selected: list[CollectedArticle] = []
        topic_buckets: dict[str, list[CollectedArticle]] = {}
        for art in articles:
            if not isinstance(art, CollectedArticle):
                continue

            body_snip = ((art.body or '').replace("\n", " ").strip()[:500])
            cache_key = f"{(art.title or '').strip()}|{(art.summary or '').strip()}|{body_snip}"
            if cache_key in cache:
                topic = cache[cache_key]
            else:
                topic = self._llm_judge_topic(art.title, art.summary, body_snip)
                if not topic:
                    topic = _normalize_title_signature(art.title or "")
                cache[cache_key] = topic

            topic_key = self._normalize_topic_key(topic)
            if topic == "정치기사제외":
                continue

            if not topic_key:
                selected.append(art)
                continue

            # 1차: 1문장 토픽 키(LLM 요약) 동일성 + 2차: 같은 주제 후보 간 LLM 직접 비교
            duplicate = False
            if topic_key in topic_buckets:
                for prev in topic_buckets[topic_key]:
                    if self._llm_is_same_topic(art, prev):
                        duplicate = True
                        break
            else:
                # topic_key가 달라서 버킷이 다르면 기존엔 비교가 안 돼 중복이 남았음.
                # 텍스트 겹침이 큰 경우에만 교차 버킷에서 1회 더 비교
                cur_tokens = _token_set(art)
                for prev in selected[-8:]:
                    if _jaccard(cur_tokens, _token_set(prev)) >= 0.55:
                        if self._llm_is_same_topic(art, prev):
                            duplicate = True
                            break

            if duplicate:
                continue

            topic_buckets.setdefault(topic_key, []).append(art)
            selected.append(art)

        return selected

    def _dedupe_articles(self, articles: list[CollectedArticle]) -> list[CollectedArticle]:
        """기사 URL + 제목 시그니처를 함께 사용해 최종 중복을 제거한다."""
        if not articles:
            return []

        selected: list[CollectedArticle] = []
        seen_urls: set[str] = set()
        seen_signatures: dict[str, int] = {}

        def _score(a: CollectedArticle) -> float:
            return (a.relevance_score or 0.0) * 1000.0 + (a.body_len or 0) + ((a.recency_score or 0.0) * 100.0)

        for a in articles:
            if not isinstance(a, CollectedArticle):
                continue
            url = (a.url or "").strip()
            sig_key = _normalize_title_signature(a.title)
            if not url:
                continue

            if url in seen_urls:
                continue

            # 제목 시그니처가 동일한 경우, 점수 높은 항목만 유지(문장 변형/중복 제목 노이즈 완화)
            if sig_key:
                prev_idx = seen_signatures.get(sig_key)
                if prev_idx is not None:
                    prev = selected[prev_idx]
                    if _score(a) > _score(prev):
                        selected[prev_idx] = a
                    continue

            seen_urls.add(url)
            seen_signatures[sig_key] = len(selected)
            selected.append(a)

        # 2차 정렬: 중요도 순 정렬 후 반환
        selected.sort(key=lambda x: (x.relevance_score or 0.0, x.body_len or 0), reverse=True)
        return selected


    def _normalize_urls_for_total(self, articles: list[CollectedArticle], filter_unresolved: bool = False) -> list[CollectedArticle]:
        seen = set()
        out: list[CollectedArticle] = []
        for a in articles:
            if not isinstance(a, CollectedArticle):
                out.append(a)
                continue
            raw = (a.url or "").strip()
            normalized = self._normalize_article_url(raw)
            if normalized:
                a.url = normalized

            if not _is_candidate_news_url(a.url):
                self.stage_counters["normalized_total"] += 1
                continue
            if filter_unresolved and not self._is_resolved_news_url(a.url):
                self.stage_counters["normalized_total"] += 1
                continue
            if a.url in seen:
                self.stage_counters["normalized_total"] += 1
                continue
            seen.add(a.url)
            out.append(a)

        # URL 정규화 이후 제목-시그니처 레벨 2차 중복도 함께 제거
        return self._dedupe_articles(out)


    def _collect_need_term_index(self, needs_payload: list[dict]) -> list[str]:
        terms: list[str] = []
        for item in needs_payload:
            if not isinstance(item, dict):
                continue
            need_text = str(item.get("need_text", "")).strip()
            if need_text:
                terms.append(need_text)
            for a in item.get("aliases", []) or []:
                a = str(a).strip()
                if a and a not in terms:
                    terms.append(a)
        return terms

    def _get_preferred_need_terms(self, needs_payload: list[dict], need_id: str | None) -> list[str]:
        if not need_id:
            return []
        for item in needs_payload:
            if not isinstance(item, dict):
                continue
            if str(item.get("need_id", "")).strip() != str(need_id).strip():
                continue
            out = []
            for v in ([item.get("need_id", ""), item.get("need_text", "")] + list(item.get("aliases", []) or [])):
                s = str(v).strip()
                if s and s not in out:
                    out.append(s)
            return out
        return [need_id]

    def _score_preselect_by_metadata(self, title: str, snippet: str, needs: list[str], preferred_need: list[str] | None = None) -> float:
        text = f"{title or ''} {snippet or ''}".lower()
        score = 0.0
        if preferred_need:
            for p in preferred_need:
                pv = str(p).lower().strip()
                if pv and pv in text:
                    score += 1.2
                    break
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

        # RSS 검색 전개: title+snippet로만 니즈 매칭
        # 원문 URL 접근성(HTTP/HEAD+GET)만 확인하고, 본문 길이 확인 단계는 생략
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

        if not self._is_url_accessible(url):
            self.stage_counters["filtered_unreachable"] += 1
            self._log(f"[collect] unreachable url={url[:90]} type={source_type}")
            return None

        # 본문 확인은 URL 접근성 검증으로 축소
        body_text = ""
        body_len = 0

        raw_title = (raw.get("title") or "").strip()
        raw_snippet = (raw.get("snippet") or "").strip()
        if not raw_title and not raw_snippet:
            self.stage_counters["fail"] += 1
            return None

        # 닫기: 제목+description로 광고/이벤트성 콘텐츠 배제
        merged_for_match = f"{raw_title} {raw_snippet}"
        if _is_marketing_content(merged_for_match, url):
            self.stage_counters["fail"] += 1
            return None

        matched_need_ids, matched_needs, matched_aliases, need_hit_score = self._collect_need_hits(merged_for_match, needs_payload)
        if not matched_need_ids:
            self.stage_counters["fail"] += 1
            return None

        published_at = self._extract_publication_time(raw)
        country = self._extract_country(raw_snippet or body_text, raw_title)

        article = CollectedArticle(
            title=raw_title,
            url=url,
            country=country,
            summary=raw_snippet,
            body=body_text,
            source="http",
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
        article.relevance_note = f"relevance_score={score:.1f}, need_hit={need_hit_score:.1f}"

        if score <= 0 and need_hit_score <= 0:
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
        # total_news 최소 목표를 안정적으로 채우기 위해 수집 단계는 버퍼를 둔다.
        target_total_candidates = min(int(self.global_candidates_cap), max(min_count, int(self.total_news_target) + 40))
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

                    need_terms = self._collect_need_term_index(needs_payload)
                    preferred_terms = self._get_preferred_need_terms(needs_payload, need_id)
                    score = self._score_preselect_by_metadata(raw.get("title") or "", raw.get("snippet") or "", need_terms, preferred_need=preferred_terms)
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
                if len(candidates) >= target_total_candidates:
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
        core_limit = limit

        # query 배치 모드로 처리
        q_batches: list[list[tuple[str, str]]] = []
        if self.query_batch_size > 0 and len(query_pairs) > self.query_batch_size:
            for i in range(0, len(query_pairs), self.query_batch_size):
                q_batches.append(query_pairs[i : i + self.query_batch_size])
        else:
            q_batches = [query_pairs]

        for batch_i, q_batch in enumerate(q_batches, start=1):
            if len(candidates) >= target_total_candidates:
                break

            self._log(f"[collect] batch start idx={batch_i}/{len(q_batches)} source=all size={len(q_batch)}")
            start_t = datetime.now().astimezone()

            # core first
            if callable(getattr(self, "search_core", None)):
                pre = stage_preselect(self.search_core, q_batch, source_type="direct", max_hits=core_limit)
                build_from_preselect(pre, "direct", start_t, source_stats)

            # google fallback: direct 검색 후 항상 보완 검색 수행
            if callable(getattr(self, "search_google", None)):
                self.stage_counters["fallback_used"] += 1
                g_pairs = q_batch[:]
                pre = stage_preselect(self.search_google, g_pairs, source_type="google", max_hits=max(25, limit // 2))
                build_from_preselect(pre, "google", start_t, source_stats)

            self._log(f"[collect] batch done idx={batch_i}/{len(q_batches)} total={len(candidates)}")

        total_candidates = candidates
        selected, total_news = self._build_total_news(total_candidates, needs_payload)
        # total_news 입력 전에 링크 정규화: google 래퍼/추적 파라미터 정리로 공통 저장 포맷 정합성 강화
        total_news = self._normalize_urls_for_total(total_news, filter_unresolved=True)
        selected = self._normalize_urls_for_total(selected, filter_unresolved=False)

        # 최종 반환/저장 전에 제목 시그니처까지 반영해 같은 내용 중복을 1건으로 압축
        total_news = self._dedupe_articles(total_news)
        selected = self._dedupe_articles(selected)

        # total_news 최소 개수 보장: 동일 필터 수준(접근 가능/URL 정합/중복 제거)을 유지한 채 후보 풀에서 보충
        target_total_news = max(1, int(self.total_news_target))
        if len(total_news) < target_total_news:
            existing_urls = {str(a.url or "").strip() for a in total_news}
            existing_sigs = {_normalize_title_signature(a.title or "") for a in total_news if (a.title or "").strip()}

            refill_pool = [a for a in total_candidates if isinstance(a, CollectedArticle) and a.extraction_status == "success_full"]
            refill_pool.sort(key=lambda x: (x.relevance_score or 0.0, x.body_len or 0), reverse=True)

            added = 0
            for cand in refill_pool:
                if len(total_news) >= target_total_news:
                    break

                norm_url = self._normalize_article_url(cand.url or "")
                if norm_url:
                    cand.url = norm_url
                if not _is_candidate_news_url(cand.url) or not self._is_resolved_news_url(cand.url):
                    continue

                u = (cand.url or "").strip()
                if not u or u in existing_urls:
                    continue
                sig = _normalize_title_signature(cand.title or "")
                if sig and sig in existing_sigs:
                    continue

                total_news.append(cand)
                existing_urls.add(u)
                if sig:
                    existing_sigs.add(sig)
                added += 1

            if added:
                total_news = self._dedupe_articles(total_news)
                self._log(f"[collect] total_news_refill added={added} target={target_total_news} now={len(total_news)}")

        self._log(f"[collect] stage_summary preselected={self.stage_counters['preselected']} built={self.stage_counters['built']} success_full={self.stage_counters['success_full']} short={self.stage_counters['short']} fail={self.stage_counters['fail']} stale={self.stage_counters['filtered_stale']} unreachable={self.stage_counters['filtered_unreachable']} normalized_total={self.stage_counters['normalized_total']} fallback={self.stage_counters['fallback_used']} short_circuit={self.stage_counters['short_circuit']}")
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
        # ==========================================================
        # DAILY FINALIZER (A안)
        # - total_news -> daily 전환 시점에서 LLM 2단계로 동일 사건 제거
        # - 중복 제거로 빠진 자리는 daily_before_topic이 아니라 total_news 풀에서 채움
        # ==========================================================

        def _rank(a: CollectedArticle) -> float:
            # 기존 점수체계를 뒤집지 않고, 이미 계산된 필드만 사용
            return float(
                (a.relevance_score or 0.0) * 1000.0
                + (a.recency_score or 0.0) * 100.0
                + (a.source_score or 1.0) * 10.0
                + (a.body_len or 0)
            )

        def _topic_cache_get(a: CollectedArticle) -> str:
            body_snip = ((a.body or "").replace("\n", " ").strip()[:500])
            key = f"{(a.title or '').strip()}|{(a.summary or '').strip()}|{body_snip}"
            if not hasattr(_topic_cache_get, "_cache"):
                _topic_cache_get._cache = {}
            cache = _topic_cache_get._cache  # type: ignore[attr-defined]
            if key in cache:
                return cache[key]
            t = self._llm_judge_topic(a.title, a.summary, body_snip)
            if not t:
                t = _normalize_title_signature(a.title or "")
            cache[key] = t
            return t

        def _token_set(a: CollectedArticle) -> set[str]:
            text = f"{a.title or ''} {a.summary or ''}"
            low = text.lower()
            tokens = set(re.findall(r"[가-힣a-z0-9]{2,}", low))
            # "젠슨 황" 같은 인명/2단어 구를 붙여 토큰으로 추가 (게이트 강화)
            for m in re.findall(r"[가-힣]{2,4}\s[가-힣]{1,2}", text):
                tokens.add(re.sub(r"\s+", "", m).lower())
            for m in re.findall(r"[A-Za-z]{2,}\s[A-Za-z]{2,}", text):
                tokens.add(re.sub(r"\s+", "", m).lower())
            return tokens

        def _jaccard(sa: set[str], sb: set[str]) -> float:
            if not sa or not sb:
                return 0.0
            inter = len(sa & sb)
            union = len(sa | sb)
            return inter / float(union) if union else 0.0

        def _build_daily_from_pool(pool: list[CollectedArticle], target_n: int) -> list[CollectedArticle]:
            # 0) 우선 점수순 정렬
            pool_sorted = sorted([x for x in pool if isinstance(x, CollectedArticle)], key=_rank, reverse=True)

            # 1) Stage-1: LLM topic label -> coarse bucket key
            buckets: dict[str, list[CollectedArticle]] = {}
            for a in pool_sorted:
                topic = _topic_cache_get(a)
                if topic == "정치기사제외":
                    continue
                topic_key = self._normalize_topic_key(topic) or _normalize_title_signature(a.title or "")
                buckets.setdefault(topic_key, []).append(a)

            # 2) Stage-2: SAME/DIFFERENT로 중복 제거하며 target_n까지 채움
            selected: list[CollectedArticle] = []
            seen_urls: set[str] = set()
            seen_sigs: set[str] = set()

            # 버킷 순서도 점수 기반으로 (대표 후보가 좋은 버킷 먼저)
            bucket_keys = sorted(
                buckets.keys(),
                key=lambda k: _rank(sorted(buckets[k], key=_rank, reverse=True)[0]),
                reverse=True,
            )

            for k in bucket_keys:
                items = sorted(buckets[k], key=_rank, reverse=True)
                reps: list[CollectedArticle] = []
                for cand in items:
                    u = (cand.url or "").strip()
                    if not u:
                        continue
                    sig = _normalize_title_signature(cand.title or "")
                    if u in seen_urls:
                        continue
                    if sig and sig in seen_sigs:
                        continue

                    dup = False

                    # 2-1) 같은 버킷 내: 대표들과 LLM SAME/DIFF
                    for rep in reps[:3]:
                        if self._llm_is_same_topic(cand, rep):
                            dup = True
                            break

                    # 2-2) 교차 버킷(놓치는 케이스 방지): 토큰 겹침이 큰 경우만 LLM SAME/DIFF
                    if not dup and selected:
                        ct = _token_set(cand)
                        for prev in selected[-10:]:
                            shared = ct & _token_set(prev)
                            sim = _jaccard(ct, _token_set(prev))
                            if sim >= 0.40 or len(shared) >= 3:
                                if self._llm_is_same_topic(cand, prev):
                                    dup = True
                                    break

                    if dup:
                        continue

                    reps.append(cand)
                    selected.append(cand)
                    seen_urls.add(u)
                    if sig:
                        seen_sigs.add(sig)

                    if len(selected) >= target_n:
                        return selected[:target_n]

            return selected[:target_n]

        # (A) 후보 풀: daily(선정 후보) + total_news(전체 풀)
        # - daily를 앞에 둬서 기존 리밸런싱 결과를 우선 반영
        pool: list[CollectedArticle] = []
        seen_pool: set[str] = set()
        for x in (list(daily) + list(total_news)):
            if not isinstance(x, CollectedArticle):
                continue
            u = (x.url or "").strip()
            if not u:
                continue
            key = f"{u}::{_normalize_title_signature(x.title or '')}"
            if key in seen_pool:
                continue
            seen_pool.add(key)
            pool.append(x)

        target_n = max(int(min_count), int(self.daily_news_target or 8))
        daily = _build_daily_from_pool(pool, target_n)

        # 권재순 사용자 중복 과다 이슈 완화: 같은 사건군 후보를 한 번 더 보수적으로 정리
        if (user.name or "").strip() == "권재순" or (user.user_code or "").strip() == "서하OXJI5394":
            def _tok(a: CollectedArticle) -> set[str]:
                s = f"{a.title or ''} {a.summary or ''}".lower()
                return set(re.findall(r"[가-힣a-z0-9]{2,}", s))

            def _jac(sa: set[str], sb: set[str]) -> float:
                if not sa or not sb:
                    return 0.0
                return len(sa & sb) / float(len(sa | sb)) if (sa | sb) else 0.0

            tightened: list[CollectedArticle] = []
            for cand in daily:
                dup = False
                ct = _tok(cand)
                for prev in tightened:
                    shared = ct & _tok(prev)
                    sim = _jac(ct, _tok(prev))
                    if sim >= 0.34 or len(shared) >= 4:
                        if self._llm_is_same_topic(cand, prev):
                            dup = True
                            break
                if not dup:
                    tightened.append(cand)
            daily = tightened

        with day_file.open("w", encoding="utf-8") as f:
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
            "filtered_unreachable_count": self.stage_counters.get("filtered_unreachable", 0),
            "normalized_total_count": self.stage_counters.get("normalized_total", 0),
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

        with day_file.open("w", encoding="utf-8") as f:
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
