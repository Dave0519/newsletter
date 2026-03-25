from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Iterable, Callable
from urllib.parse import parse_qs, urlparse, urlencode
import re
import html as _html

import requests

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None

from .models import CollectedArticle, NewsletterEntry, UserProfile
from .utils import ensure_dir
from .llm_text_utils import extract_hashtags, summarize_ko, summarize_core_ko, rewrite_title, _llm


def _load_template(template_path: Path) -> str:
    return Path(template_path).read_text(encoding="utf-8")


def _extract_block(text: str, start: str, end: str) -> str:
    a = text.find(start)
    b = text.find(end, a)
    if a == -1 or b == -1:
        raise ValueError(f"block {start}/{end} not found")
    return text[a + len(start) : b]


def _replace_block(text: str, start: str, end: str, replacement: str) -> str:
    a = text.find(start)
    b = text.find(end, a)
    if a == -1 or b == -1:
        raise ValueError(f"block {start}/{end} not found")
    return text[:a] + replacement + text[b + len(end) :]


def _safe_esc(s: str) -> str:
    return (s or "").replace("{{", "{").replace("}}", "}")


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


class WritingAgent:
    """수집 데이터 -> 공식 템플릿 변환 에이전트."""

    def __init__(self, template_path: Path, log_dir: Path, fetch_body: Callable[[str], str] | None = None, resolve_url: Callable[[str], str | None] | None = None):
        self.template_path = Path(template_path)
        self.log_dir = Path(log_dir)
        self.fetch_body = fetch_body
        self.resolve_url = resolve_url
        self.trace_enabled = os.getenv("CLUE_TRACE", "").lower() in {"1", "true", "yes", "on"}
        self.trace_root = Path(os.getenv("CLUE_TRACE_DIR", self.log_dir / "trace"))

    def _trace_path(self, user: UserProfile, name: str) -> Path:
        return self.trace_root / str((user.name or "user").replace(" ", "_")).replace("..", "_") / "writing" / name

    def _append_trace(self, path: Path, payload: dict) -> None:
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass


    def _emit_stage(
        self,
        user: UserProfile,
        issue_number: int | None,
        stage_id: str,
        status: str,
        *,
        in_count: int = 0,
        out_count: int = 0,
        elapsed_ms: int | None = None,
        extra: dict | None = None,
    ) -> None:
        if not self.trace_enabled:
            return
        rec = {
            "ts": datetime.now().astimezone().isoformat(),
            "component": "writing",
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
            f"[write_stage] user={user.user_code} issue={int(issue_number or 0)} "
            f"stage={stage_id} status={status} in={in_count} out={out_count}"
            + (f" elapsed_ms={elapsed_ms}" if elapsed_ms is not None else ""),
            flush=True,
        )

    def _clean_title(self, title: str) -> str:
        t = (title or "").strip()
        if not t:
            return t

        # source suffix 제거: e.g. "title | Source" / "title - Source"
        for sep in [" | ", " - ", " – ", " — ", " : "]:
            if sep in t:
                left, right = t.rsplit(sep, 1)
                # source 후보에는 보통 회사명/매체명이 오므로 길고 짧은 경우만 제거
                if right and 2 <= len(right) <= 40:
                    t = left.strip()
                    break
        # 괄호/브라켓 뒤쪽 부가 정보 절단
        for pat in [r"\s*\(.*\)$", r"\s*\[.*\]$"]:
            t = re.sub(pat, "", t).strip()
        return t

    def _normalize_title_candidate(self, title: str) -> str:
        t = (title or "").strip()
        t = re.sub(r"\s+", " ", t)
        t = _html.unescape(t)

        # 기사 페이지 breadcrumbs/메타 꼬리 제거
        t = re.sub(r"\s*<\s*(?:기업|정치|경제|국제|사회|생활|IT|FOCUS|오피니언|기사본문|반도체|디스플레이).*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*<\s*[^<]{0,50}$", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*-\s*(?:전자신문|인공지능신문|테크수다|TechCrunch|Reuters|Bloomberg|SBS|연합뉴스)\s*$", "", t, flags=re.IGNORECASE)

        # 깨진 괄호/브라켓 마무리
        t = re.sub(r"[\]\)>]+$", "", t).strip()
        t = t.strip("-:| ")

        if t.endswith("…"):
            t = t[:-1].strip()
        return t[:140]

    def _build_need_hashtags(self, collected: list[CollectedArticle], user: UserProfile | None = None, max_n: int = 5) -> str:
        needs: list[str] = []
        seen: set[str] = set()

        if user is not None:
            for interest in user.interests or []:
                s = str(interest or "").strip()
                key = s.lower()
                if s and key not in seen:
                    seen.add(key)
                    needs.append(s)
                    if len(needs) >= max_n:
                        break

        if len(needs) < max_n:
            for c in collected:
                for n in getattr(c, 'matched_needs', []) or []:
                    s = str(n).strip()
                    key = s.lower()
                    if s and key not in seen:
                        seen.add(key)
                        needs.append(s)
                        if len(needs) >= max_n:
                            break
                if len(needs) >= max_n:
                    break

        if not needs:
            return ""
        return " ".join([f"#{n.replace(' ', '')}" for n in needs[:max_n]])

    def _contains_korean(self, text: str) -> bool:
        return bool(re.search(r"[가-힣]", text or ""))

    def _force_korean(self, text: str, fallback_on_fail: str = "") -> str:
        t = (text or "").strip()
        if t and self._contains_korean(t):
            return t
        return fallback_on_fail

    def _clean_summary_text(self, text: str) -> str:
        if not text:
            return ""
        t = re.sub(r"\r\n|\r", " ", str(text))
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"\u3000", " ", t)
        t = _normalize_whitespace(t)

        # 1) 명백한 UI/메타데이터 패턴 제거
        noise_patterns = [
            r"로그인",
            r"회원가입",
            r"공유하기",
            r"글자크기 설정",
            r"전체메뉴",
            r"검색창",
            r"이용약관",
            r"개인정보취급방침",
            r"구독신청",
            r"게시판",
            r"뉴스검색",
            r"가상 키워드",
            r"많이 본 뉴스 기사의",
            r"추천 검색어",
            r"많이 본 뉴스",
            r"발행일",
            r"송고",
            r"댓글",
            r"서비스",
            r"주소\s*\S*",
            r"전화번호\s*\S*",
            r"제보는 언제든 환영",
            r"English Family Site",
            r"파이낸셜뉴스>",
            r"검색 English",
            r"기사 검색",
            r"기자\s*\S*\s*연락",
            r"이전 기사",
            r"다음 기사",
            r"다음은",
            r"끝까지 보기",
            r"페이스북 X",
            r"트위터",
            r"메일 URL 복사",
            r"작게 보통 크게",
            r"전체\s*서비스",
            r"이용하기",
            r"저작권\s*©",
            r"무단\s*전재",
            r"재배포\s*금지",
            r"AI\s*학습\s*이용\s*금지",
            r"최초\s*작성\s*시간",
            r"기사제목",
            r"구독\s*\w*신청",
        ]
        for p in noise_patterns:
            t = re.sub(p, " ", t, flags=re.IGNORECASE)

        # 2) 헤더 날짜/기자/입력 라인 제거
        t = re.sub(r"(?:^|\s)(?:입력|작성일|작성\s*일자|입력\s*:\s*)\s*[:\d/.\-\s년월일시분초]+(?=\s|$)", " ", t)
        t = re.sub(r"\d{4}/\d{1,2}/\d{1,2}\s*\d{1,2}:\d{2}(?:\s*\d{1,2}:\d{2})?(?:\s*\(.*?\))?", " ", t)
        t = re.sub(r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일", " ", t)
        t = re.sub(r"서울특별시\S*", " ", t)
        t = re.sub(r"\b\w+(?:구|동)\b\s*\d{1,5}[^\s,]{0,20}", " ", t)

        # 3) 사이트/메타 괄호/브래킷 제거
        t = re.sub(r"\[.*?\]", " ", t)
        t = re.sub(r"\(.*?\)", " ", t)
        t = re.sub(r"\s+", " ", t).strip()

        # 4) sentence-level로 노이즈 문장 삭제
        parts = [x.strip() for x in re.split(r"(?<=[\.。!?！？])\s+", t) if x.strip()]
        out = []
        bad_keywords = [
            "주소", "전화번호", "구독", "개인정보", "이용약관", "입력", "추천 검색어", "검색", "공유", "로그인",
            "회원가입", "뉴스검색", "댓글", "서비스", "파이낸셜뉴스", "동아일보", "헤드라인"
        ]
        for p in parts:
            if len(p) < 12:
                continue
            if any(k in p for k in bad_keywords):
                continue
            # 주소/출처 블록류 삭제
            if re.search(r"[\[\(]?(?:서울특별시|주소|발행일자|출처|기자|구독)[:\s].*", p):
                continue
            if re.match(r"^[^가-힣A-Za-z\d]{1,3}", p):
                continue
            out.append(p)

            if len(out) >= 40:
                break

        if not out:
            return t
        return self._normalize_ellipsis(". ".join(out).strip().rstrip("."))

    def _trim_body_prefix(self, text: str, title: str) -> str:
        if not text or not title:
            return text

        t = (text or "").strip()
        cleaned_title = self._clean_title(title).strip()
        if not cleaned_title:
            return text

        # keep only article body start if title appears at head
        raw_pos = t.find(cleaned_title)
        if raw_pos > 0 and raw_pos < max(20, len(cleaned_title) * 2):
            t = t[raw_pos:]

        # remove known metadata token trails
        for marker in ["입력", "공유하기", "글자크기 설정", "제보는 언제든 환영", "주소", "전화번호"]:
            m = t.find(marker)
            if 0 <= m < 120:
                candidate = t[m + len(marker):].strip()
                if candidate:
                    t = candidate

        # remove leading header-like fragments
        t = re.sub(r"^\S*\s*(?:입력|작성일|작성\s*일자)\s*[:\d/.\-\s년월일시분초]+", "", t).strip()
        return t

    def _resolve_summary_source(self, c: CollectedArticle, readable_body: str) -> str:
        body = (readable_body or "").strip()
        if body:
            cleaned = self._clean_summary_text(body)
            return self._trim_body_prefix(cleaned, c.title)
        fallback_body = (c.body or "").strip()
        if fallback_body:
            cleaned = self._clean_summary_text(fallback_body)
            return self._trim_body_prefix(cleaned, c.title)
        return ""


    def _strip_html(self, s: str) -> str:
        if not s:
            return ""
        if BeautifulSoup is not None:
            try:
                soup = BeautifulSoup(s, "html.parser")
                for bad in soup(["script", "style", "noscript"]):
                    bad.decompose()
                return _normalize_whitespace(_html.unescape(soup.get_text(" ")))
            except Exception:
                pass
        t = re.sub(r"<[^>]+>", " ", s)
        return _normalize_whitespace(_html.unescape(t))


    def _strip_summary_noise(self, text: str) -> str:
        t = (text or "").replace("…", "").replace("...", "").replace(" • ", " ")
        t = re.sub(r"\s*\([^)]*\)\s*\[[^\]]*\]\s*", " ", t)
        # remove provider tail snippets commonly leaked from news pages
        t = re.sub(r"[^.?!…]*많이\s*본\s*뉴스\s*기사의\s*키워드를\s*수집하여\s*선정하였습니다", "", t)
        t = re.sub(r"\s*가\s*작게\s*가\s*보통\s*가\s*크게\s*", " ", t)
        t = re.sub(r"\s*:\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}\s*\d{1,2}:\d{2}.*", " ", t)
        t = re.sub(r"[\(\[].*?(?:전자신문|IT 이슈플러스|회사소개|시사용어|이벤트|행사문의|고객센터|광고|뉴스\s*속보|뉴스검색|뉴스\s*검색)[^\)\]]*[\)\]]", " ", t)
        t = re.sub(r"\s*[가-힣]+=뉴시스\s+.*", " ", t)
        return self._normalize_ellipsis(_normalize_whitespace(t))

    def _llm_request(self, prompt: str, max_tokens: int = 800, temperature: float = 0.2) -> str:
        # LLM 단일 경로: 텍스트 생성 실패시 빈 문자열 반환
        return _llm(prompt, max_tokens=max_tokens, temp=temperature)

    def _llm_generate_title_from_body(self, title: str, body: str) -> str:
        base = (title or "").strip()
        if not base:
            return ""
        if body:
            return rewrite_title(base, body, max_chars=75)
        return ""

    def _llm_summary_from_body(self, title: str, body: str, regenerate_hint: str = "", line_count: int = 7) -> str:
        source = (body or "").strip()
        if not source:
            return ""
        # 모든 article_summary/core_description은 LLM 우선, 한국어로 작성
        return summarize_ko(source, title=title or "", sentence_count=line_count)

    def _llm_judge_summary(self, title: str, source: str, summary: str):
        # Conservative fallback: rely on source alignment check
        ok = bool(summary) and self._assert_source_alignment(summary, source)
        return ok, "" if ok else "요약 근거 정합성 점검 실패"

    def _llm_refine_from_source(self, kind: str, text: str, source: str, max_lines: int = 5, linebreak: bool = True) -> str:
        src = (source or "").strip()
        cur = (text or "").strip()
        if not src or not cur:
            return cur
        fmt = "문장당 한 줄로 줄바꿈" if linebreak else "줄바꿈 없이 한 문단"
        prompt = (
            f"아래 {kind} 초안을 기사 원문 근거로 교정해라.\n"
            "목표: 메타/UI/광고 문구 제거, 원문 사실만 유지, 한국어만 사용.\n"
            "원문에 없는 추측/전망/평가 금지, 내용 왜곡 금지.\n"
            "출력은 절대 '...' 또는 '…'를 쓰지 말고, 문장이 끊긴 채로 끝내지 말 것.\n"
            "문장 끝은 마침표/물음표/느낌표로 마감하고, 필요하면 문장만 재작성할 것.\n"
            f"출력 형식: 최대 {max_lines}문장, {fmt}, 설명/라벨 없이 결과문만 출력.\n\n"
            f"[원문]\n{src[:9000]}\n\n"
            f"[{kind} 초안]\n{cur[:3000]}"
        )
        out = (self._llm_request(prompt, max_tokens=700, temperature=0.1) or "").strip()
        return out or cur

    def _korean_country(self, country: str) -> str:
        c = (country or "").strip().upper()
        mapping = {
            "KR": "한국",
            "KOREA": "한국",
            "US": "미국",
            "USA": "미국",
            "GLOBAL": "글로벌",
            "CN": "중국",
            "JP": "일본",
            "TW": "대만",
            "EU": "유럽",
            "HK": "홍콩",
            "IN": "인도",
            "DE": "독일",
            "FR": "프랑스",
            "UK": "영국",
        }
        return mapping.get(c, c or "글로벌")

    def _resolve_url(self, url: str) -> str:
        return self._resolve_source_url(url)

    def _canonicalize_url(self, url: str) -> str:
        try:
            p = urlparse((url or "").strip())
            if not p.scheme or not p.netloc:
                return (url or "").strip()
            drop_prefix = ("utm_", "fbclid", "gclid", "oc", "ref", "ref_src")
            q = parse_qs(p.query, keep_blank_values=False)
            kept = []
            for k, vals in q.items():
                lk = (k or "").strip().lower()
                if lk in drop_prefix or any(lk.startswith(x) for x in drop_prefix):
                    continue
                for v in vals:
                    kept.append((k, v))
            query = urlencode(kept, doseq=True)
            clean = p._replace(query=query, fragment="")
            return clean.geturl().rstrip("?")
        except Exception:
            return (url or "").strip()

    def _resolve_source_url(self, url: str) -> str:
        raw = (url or "").strip()
        if not raw:
            return ""
        if "news.google.com" not in raw:
            return self._canonicalize_url(raw)

        try:
            parsed = urlparse(raw)
            qs = parse_qs(parsed.query)
            if qs.get("url") and qs["url"][0]:
                return self._canonicalize_url(qs["url"][0])
        except Exception:
            pass

        candidate = raw
        if self.resolve_url:
            try:
                candidate = self.resolve_url(raw) or raw
            except Exception:
                candidate = raw

        return self._canonicalize_url(candidate)

    def _ensure_korean_summary_lines(self, summary: str, max_lines: int = 5) -> str:
        if not summary:
            return summary
        lines = [x.strip() for x in summary.split("\n") if x.strip()]
        if len(lines) >= 2:
            return "\n".join(lines[:max_lines])
        parts = [p.strip() for p in summary.split(". ") if p.strip()]
        out = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if not p.endswith("."):
                p += "."
            out.append(p)
            if len(out) >= max_lines:
                break
        return "\n".join(out)

    def _extract_summary_candidates(self, source: str) -> list[str]:
        text = self._clean_summary_text(source or "")
        if not text:
            return []

        chunks: list[str] = []
        # sentence split
        for p in re.split(r"(?<=[\.!?！？])\s+", text):
            p = re.sub(r"\s+", " ", (p or "").strip())
            if p and len(p) >= 12:
                chunks.append(p)

        if not chunks:
            chunks = [x.strip() for x in re.split(r"[\n\r]+", text) if x.strip() and len(x.strip()) >= 12]

        # metadata-like / publisher lines 제거
        blocked = {
            "입력 ", "작성일", "수정", "주소", "서울특별시", "구독", "동아일보", "기사검색", "뉴스검색", "많이 본 뉴스 기사의 키워드를 수집하여 선정하였습니다", "가 작게 가 보통 가 크게", "발행일", "키워드를 수집하여 선정하였습니다", "뉴스 검색", "전자신문", "이천=뉴시스", "뉴시스", "회사소개", "시사용어", "이벤트", "행사문의", "고객센터", "고충처리", "기자", "©",
        }
        out: list[str] = []
        for c in chunks:
            if any(b in c for b in blocked):
                continue
            out.append(c)
            if len(out) >= 20:
                break
        return out


    def _ensure_min_summary_lines(self, summary: str, source: str, min_lines: int = 6, max_lines: int = 7) -> str:
        text = (summary or "").strip()
        lines = [x.strip() for x in text.split("\n") if x.strip()]
        if len(lines) >= min_lines:
            return "\n".join(lines[:max_lines])

        for c in self._extract_summary_candidates(source):
            if c and c not in lines:
                lines.append(c)
            if len(lines) >= max_lines:
                break

        if len(lines) < min_lines:
            fallback = self._strip_html(source or "")
            fallback = re.sub(r"\s+", " ", fallback).strip()
            if fallback:
                chunk = 140
                while len(lines) < min_lines and len(fallback) > 30:
                    piece = fallback[:chunk].strip()
                    if piece and piece not in lines:
                        lines.append(piece)
                    fallback = fallback[chunk:].strip()

        return "\n".join(lines[:max_lines])

    def _normalize_practical_lines(self, text: str, max_lines: int = 5) -> str:
        if not text:
            return text
        lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(?:[0-9]+[\)\.]|[가-힣]+\)|[①-⑳]|[#•\-*]\s*|첫째|둘째|셋째|넷째|다섯째|여섯째)\s*[:\-]?\s*", "", line)
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            lines.append(line)
            if len(lines) >= max_lines:
                break
        return "\n".join(lines)

    def _normalize_numeric_units(self, text: str) -> str:
        return re.sub(r"[\s,]", "", (text or "")).lower()

    def _extract_numeric_tokens(self, text: str) -> set[str]:
        if not text:
            return set()

        tokens: set[str] = set()
        src = text
        for m in re.findall(r"\b(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:%|\$|usd|eur|원|만원|억원|억|조|달러|유로|위안|위안화|엔|kg|톤|개|건|명|년|개월|주|일|시간|분|초|%|bps|mbps|gbps)?\b", src, flags=re.IGNORECASE):
            tokens.add(re.sub(r"[\s,]", "", m))
        for m in re.findall(r"20\d{2}\s*년", src):
            tokens.add(re.sub(r"\s+", "", m))

        # fallback 숫자 단독 추출
        for m in re.findall(r"\d+", src):
            tokens.add(m)
        return {self._normalize_numeric_units(x) for x in tokens if x}
    def _find_unmatched_numbers(self, text: str, source: str) -> set[str]:
        source_tokens = self._extract_numeric_tokens(source)
        out: set[str] = set()
        for n in self._extract_numeric_tokens(text):
            if n and n not in source_tokens:
                out.add(n)
        return out

    def _to_complete_sentences(self, text: str, max_lines: int = 5) -> str:
        t = self._normalize_ellipsis((text or "")).replace("\r", "").strip()
        if not t:
            return ""

        replaced = re.sub(r"\s+", " ", t)
        segs = re.split(r"(?<=[\.。!?！？])\s+", replaced)
        if not segs:
            segs = [replaced]

        out = []
        for s in segs:
            s = self._normalize_ellipsis(s.strip())
            if not s:
                continue
            if not re.search(r"[\.。!?！？]$", s):
                s += "."
            out.append(s)
            if len(out) >= max_lines:
                break

        return "\n".join(out[:max_lines])

    def _normalize_ellipsis(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return t
        # 문장 말미 생략 표현(...) 제거 후 정리
        t = re.sub(r"\.\.\.|\.\.{2,}|…", "", t)
        t = re.sub(r"\s{2,}", " ", t)
        t = re.sub(r"\.\s*,", ",", t)
        t = t.strip()
        return t

    def _to_complete_summary_lines(self, text: str, max_lines: int = 7) -> str:
        t = self._normalize_ellipsis((text or "")).replace("\r", "").strip()
        if not t:
            return ""

        replaced = re.sub(r"\s+", " ", t)
        segs = re.split(r"(?<=[\.。!?！？])\s+", replaced)
        if not segs:
            segs = [replaced]

        out = []
        for s in segs:
            s = self._normalize_ellipsis(s.strip())
            if not s:
                continue
            if not re.search(r"[\.。!?！？]$", s):
                s += "."
            out.append(s)
            if len(out) >= max_lines:
                break

        return "\n".join(out[:max_lines])

    def _compact_2sentence_summary(self, text: str, max_sentences: int = 2) -> str:
        t = self._normalize_ellipsis((text or "")).replace("\r", " ").strip()
        if not t:
            return ""

        # preserve explicit newlines first, otherwise split by sentence
        lines = [x.strip() for x in t.split("\n") if x.strip()]
        if len(lines) >= 2:
            selected = lines
        else:
            selected = [x.strip() for x in re.split(r"(?<=[\.。!?！？])\s+", t) if x.strip()]
            if not selected:
                selected = [seg.strip() for seg in re.split(r"(?<=다)\s*", t) if seg.strip()]

        out = []
        for s in selected:
            s = self._normalize_ellipsis(s.strip())
            if not s:
                continue
            if not re.search(r"[\.。!?！？]$", s):
                s += "."
            out.append(s)
            if len(out) >= max_sentences:
                break

        return " ".join(out).strip()

    def _repair_by_source(self, text: str, source: str, prompt_suffix: str, max_lines: int = 3) -> str:
        if not source:
            return text

        hint = (
            "다음 텍스트를 본문에서 직접 확인 가능한 내용으로만 보강/정제해줘. "
            f"숫자/단위는 본문 표현을 그대로 유지하고 임의로 바꾸지 마.\n{prompt_suffix}\n"
            f"[본문]\n{source[:16000]}"
        )

        out = self._llm_request(
            hint + f"\n\n[현재 텍스트]\n{text}",
            max_tokens=max(220, 250 * max_lines),
            temperature=0.1,
        )
        if not out:
            return text
        candidate = self._to_complete_sentences(out, max_lines=max_lines)
        if not candidate:
            return text
        return candidate

    def _assert_source_alignment(self, text: str, source: str) -> bool:
        # 수치/단위 기준 1차 검증 (정확한 의미 매칭은 LLM 검수에서 처리)
        unmatched = self._find_unmatched_numbers(text, source)
        return len(unmatched) == 0


    def _judge_country_by_title(self, title: str) -> str:
        """Use LLM to classify article title into one of KR/US/CN/TW/GLOBAL."""
        return self._judge_country_from_context(title=title, body_text="", summary="")

    def _judge_country_by_rules(self, title: str, body: str, summary: str, source_url: str) -> str:
        text = f"{(title or '')} {(summary or '')} {(body or '')}"
        url = (source_url or "").lower()

        if re.search(r"\b(SAMSUNG|SK|LG|HYUNDAI|KOREA|KOREAN)\b", text, re.IGNORECASE):
            return "KR"
        if re.search(r"\b(USA|US|NVIDIA|OPENAI|AMD|AMAZON|GOOGLE|APPLE|TESLA|MICROSOFT|NASA|REUTERS)\b", text, re.IGNORECASE):
            return "US"
        if re.search(r"\b(CHINA|CN|HUAWEI|TENCENT|ALIBABA|BAIDU|XIAOMI)\b", text, re.IGNORECASE):
            return "CN"
        if re.search(r"\b(TAIWAN|TSMC)\b", text, re.IGNORECASE):
            return "TW"

        if "ft.com" in url or "wsj.com" in url or "reuters.com" in url or "bloomberg.com" in url or "cnn.com" in url:
            return "US"
        if "ftchinese.com" in url or "xinhua" in url or "chinanews" in url:
            return "CN"
        if "cna.com.tw" in url or "tsmc" in url:
            return "TW"

        if re.search(r"\.kr($|/)", url):
            return "KR"
        return "GLOBAL"

    def _judge_country_from_context(
        self,
        title: str,
        body_text: str,
        summary: str = "",
        source_url: str = "",
    ) -> str:
        title_text = (title or "").strip()
        body = (body_text or "").strip()[:3200]
        summ = (summary or "").strip()[:1600]
        url = (source_url or "").strip()

        combined = "\n".join([x for x in (title_text, summ, body) if x]).strip()

        if combined:
            out = (
                self._llm_request(
                    "classify article country into KR/US/CN/TW/GLOBAL\n" + combined + ("\n" + url if url else ""),
                    max_tokens=16,
                    temperature=0.0,
                )
                or ""
            ).strip().upper()

            if out:
                for tok in re.split(r"[\s,;/|]+", out):
                    t = tok.strip().upper().strip(".\n ")
                    if t in {"KR", "US", "CN", "TW", "GLOBAL"}:
                        return t

        return self._judge_country_by_rules(title_text, body, summ, url)

    def _coerce_country_code(self, country: str) -> str:
        c = (country or "").strip().upper()
        mapping = {
            "KOREA": "KR",
            "KOR": "KR",
            "KOREAN": "KR",
            "한국": "KR",
            "미국": "US",
            "USA": "US",
            "AMERICA": "US",
            "CHINA": "CN",
            "CHINESE": "CN",
            "중국": "CN",
            "TAIWAN": "TW",
            "대만": "TW",
            "글로벌": "GLOBAL",
            "GLOBAL": "GLOBAL",
            "JP": "GLOBAL",
            "일본": "GLOBAL",
            "JAPAN": "GLOBAL",
            "EU": "GLOBAL",
            "유럽": "GLOBAL",
            "HK": "GLOBAL",
            "홍콩": "GLOBAL",
            "IN": "GLOBAL",
            "인도": "GLOBAL",
            "DE": "GLOBAL",
            "독일": "GLOBAL",
            "FR": "GLOBAL",
            "프랑스": "GLOBAL",
            "UK": "GLOBAL",
            "영국": "GLOBAL",
        }
        return mapping.get(c, c or "GLOBAL") if c in mapping else c if c in {"KR", "US", "CN", "TW", "GLOBAL"} else "GLOBAL"


    def _extract_readable_text(self, html: str) -> str:
        if not html:
            return ""
        try:
            if BeautifulSoup is not None:
                soup = BeautifulSoup(html, "html.parser")
                for bad in soup(["script", "style", "noscript"]):
                    bad.decompose()
                text = soup.get_text(" ")
                return self._clean_summary_text(_normalize_whitespace(_html.unescape(text)))
        except Exception:
            pass
        text = re.sub(r"<[^>]+>", " ", html)
        return self._clean_summary_text(_normalize_whitespace(_html.unescape(text)))


    def _load_remote_body(self, url: str) -> str:
        if not self.fetch_body or not url:
            return ""
        try:
            text = self.fetch_body(url)
            return text or ""
        except Exception:
            return ""

    def _extract_title_from_html(self, html: str, fallback_url: str = "") -> str:
        if html and BeautifulSoup is not None:
            try:
                soup = BeautifulSoup(html, "html.parser")
                for node in (soup.find("meta", property="og:title"), soup.find("meta", attrs={"name": "twitter:title"})):
                    if node is not None:
                        content = (node.get("content") or "").strip()
                        if content:
                            return self._normalize_title_candidate(self._clean_title(content))
                t = (soup.title.string if soup.title else "") or ""
                if t:
                    return self._normalize_title_candidate(self._clean_title(t))
                h1 = soup.find("h1")
                if h1:
                    return self._normalize_title_candidate(self._clean_title(h1.get_text(" ", strip=True)))
            except Exception:
                pass

        # fallback: if stripped text was passed (adapter text mode), pull full html by URL
        if fallback_url and BeautifulSoup is not None:
            try:
                r = requests.get(fallback_url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                for node in (soup.find("meta", property="og:title"), soup.find("meta", attrs={"name": "twitter:title"})):
                    if node is not None:
                        content = (node.get("content") or "").strip()
                        if content:
                            return self._normalize_title_candidate(self._clean_title(content))
                t = (soup.title.string if soup.title else "") or ""
                if t:
                    return self._normalize_title_candidate(self._clean_title(t))
                h1 = soup.find("h1")
                if h1:
                    return self._normalize_title_candidate(self._clean_title(h1.get_text(" ", strip=True)))
            except Exception:
                return ""
        return ""

    def _ordered_entry_payloads(self, payloads: list[tuple[NewsletterEntry, dict[str, str]]]) -> list[tuple[NewsletterEntry, dict[str, str]]]:
        grouped: dict[str, list[tuple[NewsletterEntry, dict[str, str]]]] = {}
        for entry, ctx in payloads:
            key = entry.country or "GLOBAL"
            grouped.setdefault(key, []).append((entry, ctx))

        ordered = []
        for c in ["KR", "US", "CN", "TW", "GLOBAL"]:
            if c in grouped:
                ordered.extend(grouped[c])
        for c, items in grouped.items():
            if c not in {"KR", "US", "CN", "TW", "GLOBAL"}:
                ordered.extend(items)
        return ordered

    def _build_summary_points(self, entries_payload: list[tuple[NewsletterEntry, dict[str, str]]], max_lines: int = 2) -> str:
        """GLOBAL SCAN 렌더링 순서를 그대로 반영해 CORE_DESCRIPTION 구성.

        - 본문 본문만 사용해 요약(요약 원본/메타데이터는 사용하지 않음)
        - 각 기사당 1~2줄
        """
        lines: list[str] = []

        # GLOBAL SCAN에서 실제 표시되는 국가 블록/기사 순서를 유지
        ordered_payloads = self._ordered_entry_payloads(entries_payload)

        for entry, ctx in ordered_payloads:
            title = (entry.title or "").strip()
            # Core description도 article_summary와 동일하게 daily_news 본문 기반 LLM 요약
            body = (ctx.get("readable_body") or "").strip()

            compact = ""
            if body:
                # core_description은 LLM 출력만 사용(본문 조각 보강 금지)
                compact = summarize_core_ko(body, title=title, sentence_count=2)
                compact = self._llm_refine_from_source("core_description", compact, body, max_lines=2, linebreak=True)
                compact = self._clean_summary_text(compact)
                compact = self._ensure_korean_summary_lines(compact, max_lines=2).strip()
                compact = self._strip_summary_noise((compact or "").strip())
                compact = self._to_complete_summary_lines(compact, max_lines=2)

            if compact:
                # 기사별 2줄 유지
                lines.append(f"• {compact.replace(chr(10), '<br/>')}")

        return "<br/><br/>".join(lines) if lines else "오늘은 본문 추출 가능한 주요 기사가 부족했습니다."


    def _grouped_country_blocks(self, articles: list[NewsletterEntry]):
        countries: dict[str, list[NewsletterEntry]] = {}
        for a in articles:
            countries.setdefault(a.country or "GLOBAL", []).append(a)
        return countries

    def _ordered_country_blocks(self, countries: dict[str, list[NewsletterEntry]]):
        # global scan block order: 한국 -> 미국 -> 중국 -> 대만 -> 글로벌
        order = ["KR", "US", "CN", "TW", "GLOBAL"]
        # keep unknowns at end in discovery order
        out = OrderedDict()
        for c in order:
            if c in countries:
                out[c] = countries[c]
        for c, arts in countries.items():
            if c not in order and c not in out:
                out[c] = arts
        return out

    def _derive_entry_payload(self, c: CollectedArticle, user: UserProfile | None = None, issue_number: int | None = None) -> tuple[NewsletterEntry, dict[str, str]]:
        if self.trace_enabled and user is not None:
            t0 = datetime.now().astimezone()
            self._emit_stage(user, issue_number, stage_id="derive_entry", status="start", in_count=1, out_count=0)
        issue_snapshot = int(issue_number or 0)
        resolved_url = self._resolve_url(c.url)
        # Writing 단계는 URL 재요청 없이 daily news 본문 기반으로만 요약한다.
        readable_body = str(c.body or "").strip()
        raw_title = self._normalize_title_candidate(self._clean_title(c.title))
        summary_source = self._resolve_summary_source(c, readable_body)

        # 제목: 기사 제목 우선 사용 (daily news 기반)
        title = raw_title or c.title or "제목 변환이 아직 완료되지 않았습니다."
        # 깨진 패턴/너무 짧은 제목은 본문 기반 재작성으로 교정
        bad_title = (
            (not title)
            or (title.strip() in {"네이버뉴스.", "네이버뉴스", "Google News.", "Google News", "google news.", "google news", "제목 변환이 아직 완료되지 않았습니다."})
            or len(title.strip()) < 6
            or ("기사본문" in title)
            or title.strip().endswith("]")
        )
        if bad_title:
            regenerated = self._llm_generate_title_from_body(title, summary_source)
            title = self._normalize_title_candidate(regenerated or raw_title or c.title or title)

        # 정책: article_title도 LLM 기반 한국어 작성 우선
        if title and not self._contains_korean(title):
            ko_title = ""
            try:
                ko_title = self._llm_request(
                    "다음 기사 제목과 본문을 바탕으로 뉴스레터용 한국어 제목 1문장으로 작성해줘. "
                    "반드시 한국어로만 작성하고, 45자 내외로 간결하게 작성해. "
                    "[주니어전자], [속보], [단독] 같은 매체/라벨 접두어는 절대 포함하지 마.\n\n"
                    f"[제목]\n{title}\n\n[본문]\n{(summary_source or '')[:1200]}",
                    max_tokens=80,
                    temperature=0.2,
                )
            except Exception:
                ko_title = ""
            ko_title = self._normalize_title_candidate((ko_title or "").strip())
            title = self._normalize_title_candidate(ko_title or title)

        title = self._to_complete_sentences(title, max_lines=1)

        # article_summary / practical은 모두 daily_news body 기반으로 각각 다른 프롬프트를 병렬 생성
        def _gen_summary() -> str:
            d = self._llm_summary_from_body(title, summary_source, line_count=5)
            if d:
                ok, reason = self._llm_judge_summary(title, summary_source, d)
                if not ok:
                    d = self._llm_summary_from_body(title, summary_source, regenerate_hint=reason, line_count=7)
            if not d:
                d = summarize_ko(summary_source, title=title, sentence_count=7)
            if not d:
                d = self._strip_html(summary_source)
            return d

        def _gen_practical() -> str:
            return self._openclaw_style_practical(
                title=title,
                summary="",
                body=summary_source,
                user=user,
                issue_number=issue_number,
            )

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_summary = ex.submit(_gen_summary)
            fut_practical = ex.submit(_gen_practical)
            description = fut_summary.result() or ""
            practical_prefetch = fut_practical.result() or ""

        if not description:
            description = "해당 기사를 본문에서 핵심 내용을 추출하지 못해 요약 텍스트가 비어 있습니다."

        description = self._ensure_korean_summary_lines(description, max_lines=5)
        description = self._ensure_min_summary_lines(description, source=summary_source, min_lines=3, max_lines=5)
        description = self._llm_refine_from_source("article_summary", description, summary_source, max_lines=5, linebreak=True)
        description = self._clean_summary_text(description)
        if not description:
            description = "해당 기사를 본문에서 핵심 내용을 추출하지 못해 요약 텍스트가 비어 있습니다."
        description = self._to_complete_summary_lines(description, max_lines=5)
        if not self._assert_source_alignment(description, summary_source):
            description = self._repair_by_source(description, summary_source, "기사 요약을 본문 근거로 3~5줄 이내로 완성형 문장으로 수정", max_lines=5)

        # 최종 메타 문구/저작권 라인 제거
        cleaned_lines: list[str] = []
        for ln in [x.strip() for x in str(description or "").splitlines() if x.strip()]:
            if re.search(r"저작권|무단\s*전재|재배포\s*금지|AI\s*학습\s*이용\s*금지|송고\s*\d", ln, re.IGNORECASE):
                continue
            cleaned_lines.append(ln)
            if len(cleaned_lines) >= 5:
                break
        description = "\n".join(cleaned_lines)
        description = self._clean_summary_text(description)
        description = self._to_complete_summary_lines(description, max_lines=5)
        if not description:
            description = "해당 기사를 본문에서 핵심 내용을 추출하지 못해 요약 텍스트가 비어 있습니다."

        entry = NewsletterEntry(
            title=title,
            summary=description,
            url=resolved_url,
            country=c.country,
            practical_implication="",
            need_category=c.need_category,
            topic=c.need_category or "AI/반도체 동향",
        )
        ctx = {
            "summary_source": summary_source,
            "readable_body": readable_body,
            "raw_title": raw_title,
            "input_summary": c.summary or "",
            "precomputed_practical": practical_prefetch,
        }
        if self.trace_enabled and user is not None:
            elapsed_ms = int((datetime.now().astimezone() - t0).total_seconds() * 1000)
            self._emit_stage(user, issue_snapshot, stage_id="derive_entry", status="done", in_count=1, out_count=1, elapsed_ms=elapsed_ms)
        return entry, ctx

    def _openclaw_style_practical(self, title: str, summary: str, body: str, user: UserProfile | None = None, issue_number: int | None = None) -> str:
        if self.trace_enabled and user is not None:
            t0 = datetime.now().astimezone()
            self._emit_stage(user, issue_number, stage_id="practical", status="start", in_count=1, out_count=0)

        if not title and not summary:
            if self.trace_enabled and user is not None:
                self._emit_stage(user, issue_number, stage_id="practical", status="skipped", in_count=1, out_count=0)
            return ""


        body_snippet = (body or "")[:8000]
        prompt = (
            "당신은 뉴스/산업 기사에서 실무적 시사점을 부드럽게 도출하는 전략 보조자입니다.\n\n"
            "아래 기사 원문을 바탕으로, SK하이닉스 실무자 관점에서 참고할 만한 가벼운 실무 시사점을 작성하세요.\n"
            "원문이 영어(또는 다국어)여도 출력은 반드시 한국어로만 작성하세요.\n\n"
            "[작성 목표]\n"
            "- 단정적인 결론이나 과도한 해석이 아니라, \"이런 관점에서도 생각해볼 수 있겠다\" 수준의 가벼운 인사이트를 제시하세요.\n"
            "- 읽는 사람이 부담 없이 받아들이되, 내부적으로 한 번 더 생각해보게 만드는 질문형 문제의식을 담으세요.\n"
            "- 정답을 제시하는 느낌보다, 실무자가 스스로 판단할 수 있도록 사고의 방향을 열어주는 느낌으로 작성하세요.\n\n"
            "[핵심 원칙]\n"
            "- 반드시 기사 원문에 나온 사실만 근거로 삼으세요.\n"
            "- 원문에 없는 정보, 외부 지식, 추측, 내부 사정 가정, 확정적 전망은 넣지 마세요.\n"
            "- '반드시', '결국', '분명히', '즉시 대응해야 한다' 같은 단정적 표현은 피하세요.\n"
            "- 위기감 조성, 과장된 해석, 지나친 전략적 비약은 금지합니다.\n"
            "- 실무자 관점에서 현실적으로 생각해볼 포인트만 가볍게 정리하세요.\n"
            "- 문체는 부드럽고 절제된 비즈니스 문체로 작성하세요.\n"
            "- 뉴스레터 본문에서 양쪽정렬로 읽힌다는 점을 고려해 문장 길이와 호흡을 균형 있게 맞추세요.\n\n"
            "[톤 가이드]\n"
            "- '~로 볼 수 있다' / '~도 생각해볼 수 있다' / '~여부를 점검해볼 필요가 있다' / '~관점에서 한 번 볼 만하다' 같은 표현을 활용하세요.\n"
            "- 훈수, 지시, 결론 선언처럼 들리지 않게 작성하세요.\n"
            "- '이 기사가 곧바로 무엇을 의미한다'보다 '이 흐름이 실무에서 어떤 질문으로 이어질 수 있는지'를 보여주세요.\n\n"
            "[출력 형식]\n"
            "- 한국어로 작성하세요.\n"
            "- 줄바꿈 없이 하나의 짧은 문단으로 작성하세요.\n"
            "- 2~4문장으로 작성하세요.\n"
            "- 기사 요약을 반복하지 말고, 시사점만 작성하세요.\n"
            "- 필요하면 마지막 문장을 가벼운 질문형으로 마무리하세요.\n"
            "- 질문은 1개 또는 2개까지만 포함하세요.\n"
            "- 질문은 압박형이 아니라 검토형이어야 합니다.\n\n"
            "[금지 사항]\n"
            "- SK하이닉스의 내부 전략을 아는 것처럼 쓰지 마세요.\n"
            "- 투자 조언처럼 쓰지 마세요.\n"
            "- 기사 사실을 넘어선 산업 전망을 단정하지 마세요.\n"
            "- 과장된 위기론/낙관론을 쓰지 마세요.\n"
            "- 'SK하이닉스는 ~해야 한다' 식의 명령형 표현을 쓰지 마세요.\n\n"
            "이제 아래 기사 원문을 바탕으로 실무 시사점을 작성하세요.\n\n"
            f"기사 원문:\n{body_snippet}"
        )
        out = self._llm_request(prompt, max_tokens=420, temperature=0.2)
        out = (out or "").strip()
        if self.trace_enabled and user is not None:
            elapsed_ms = int((datetime.now().astimezone() - t0).total_seconds() * 1000)
            self._emit_stage(user, issue_number, stage_id="practical", status="done", in_count=1, out_count=1 if out else 0, elapsed_ms=elapsed_ms)
        if not out:
            return ""
        out = self._normalize_practical_lines(self._ensure_korean_summary_lines(out, max_lines=3), max_lines=3)
        return out


    def _attach_practical_implications(self, entries: list[NewsletterEntry], contexts: list[dict[str, str]], user: UserProfile | None = None, issue_number: int | None = None) -> list[NewsletterEntry]:
        if self.trace_enabled and user is not None:
            t0 = datetime.now().astimezone()
            self._emit_stage(user, issue_number, stage_id="attach_practical", status="start", in_count=len(entries), out_count=0)
        out: list[NewsletterEntry] = []
        for e, ctx in zip(entries, contexts):
            practical_source = (ctx.get("summary_source") or "").strip()
            practical_body = (ctx.get("readable_body") or "").strip()

            practical = (ctx.get("precomputed_practical") or "").strip()
            if not practical:
                practical = self._openclaw_style_practical(
                    title=e.title,
                    summary="",
                    body=(practical_body or practical_source),
                    user=user,
                    issue_number=issue_number,
                )
            practical = self._llm_refine_from_source("실무 시사점", practical, practical_body or practical_source, max_lines=3, linebreak=False)
            if not self._assert_source_alignment(practical or e.summary, practical_source or practical_body):
                practical = self._repair_by_source(practical or e.summary, practical_source or practical_body, "실무 시사점을 본문 근거로 3문장 이내로 정리", max_lines=3)
            e.practical_implication = practical
            out.append(e)
        if self.trace_enabled and user is not None:
            self._emit_stage(user, issue_number, stage_id="attach_practical", status="done", in_count=len(entries), out_count=len(out), elapsed_ms=int((datetime.now().astimezone() - t0).total_seconds() * 1000))
        return out

    def _derive_entry(self, c: CollectedArticle) -> NewsletterEntry:
        entry, _ = self._derive_entry_payload(c)
        return entry


    def _build_trace_id(self, user: UserProfile, issue_number: int | None, article_count: int) -> str:
        issue = str(issue_number or 0).zfill(3)
        t = datetime.now().strftime("%Y%m%d%H%M%S")
        base = f"{t}|{user.user_code}|{issue}|{article_count}"
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
        return f"clue_{t}_{user.user_code}_{issue}_{digest}"

    def _inject_trace_comment(self, html: str, user: UserProfile, issue_number: int | None, article_count: int) -> str:
        trace_id = self._build_trace_id(user, issue_number, article_count)
        marker = (
            f"<!-- CLUE_INTERNAL_TRACE user_code={user.user_code} user={user.name} "
            f"issue={issue_number or 0} articles={article_count} trace_id={trace_id} -->"
        )
        if "</body>" in html:
            return html.replace("</body>", marker + "\n</body>")
        return html + marker

    def compose_html(self, user: UserProfile, collected: list[CollectedArticle], issue_number: int | None = None) -> str:
        if self.trace_enabled:
            t0 = datetime.now().astimezone()
            self._emit_stage(user, issue_number, stage_id="compose", status="start", in_count=len(collected), out_count=0)
        if not collected:
            raise ValueError("No collected article")

        entries_payload = [self._derive_entry_payload(c, user=user, issue_number=issue_number) for c in collected]
        entries = [entry for entry, _ in entries_payload]
        entries = self._attach_practical_implications(entries, [ctx for _, ctx in entries_payload], user=user, issue_number=issue_number)

        template = _load_template(self.template_path)
        template = template.replace("{{ISSUE_DATE}}", datetime.now().strftime("%Y. %m. %d"))
        template = template.replace("{{ISSUE_NUMBER}}", str(issue_number or 0).zfill(3))
        template = template.replace("{{SERIAL_NUMBER}}", user.user_code)
        template = template.replace("{{BRAND_MARK}}", "SK HYNIX")

        # header tags: use top needs directly from collected list
        # if insufficient needs, fallback to text keyword extraction
        need_tags = self._build_need_hashtags(collected, user=user, max_n=5)
        template = template.replace("{{NEEDS_HASHTAGS}}", need_tags)

        # Country/article blocks
        # Global scan block mapping is LLM-driven from article title+summary+body context.
        for e, ctx in entries_payload:
            judged = self._judge_country_from_context(
                title=e.title,
                body_text=(ctx.get("readable_body") or ctx.get("summary_source") or ""),
                summary=e.summary,
                source_url=e.url,
            )
            e.country = self._coerce_country_code(judged)

        # summary section follows GLOBAL SCAN(국가 블록) 렌더 순서
        # CORE_DESCRIPTION는 본문 기반 6줄 이내로 구성
        summary_lines = self._build_summary_points(entries_payload, max_lines=2)
        summary_block = summary_lines
        template = template.replace("{{CORE_DESCRIPTION}}", summary_block)

        countries = self._ordered_country_blocks(self._grouped_country_blocks(entries))
        row_t = _extract_block(template, "{{#COUNTRIES}}", "{{/COUNTRIES}}")
        art_t = _extract_block(row_t, "{{#ARTICLES}}", "{{/ARTICLES}}")
        rows = []
        for country, items in countries.items():
            row = row_t.replace("{{COUNTRY_NAME}}", self._korean_country(country))
            ars = []
            for it in items:
                r = art_t
                r = r.replace("{{ARTICLE_TITLE}}", _safe_esc(it.title))
                r = r.replace("{{ARTICLE_SUMMARY}}", _safe_esc(it.summary))
                r = r.replace("{{ARTICLE_PRACTICAL_IMPLICATION}}", _safe_esc(it.practical_implication))
                r = r.replace("{{ARTICLE_LINK}}", it.url)
                ars.append(r)
            row = _replace_block(row, "{{#ARTICLES}}", "{{/ARTICLES}}", "\n".join(ars))
            rows.append(row)

        template = _replace_block(template, "{{#COUNTRIES}}", "{{/COUNTRIES}}", "\n".join(rows))

        # placeholders that may be introduced later
        template = template.replace("{{GLOBAL_SCAN_INTRO}}", "")
        if self.trace_enabled:
            self._emit_stage(user, issue_number, stage_id="compose", status="done", in_count=len(collected), out_count=len(entries), elapsed_ms=int((datetime.now().astimezone() - t0).total_seconds() * 1000))
        return self._inject_trace_comment(template, user=user, issue_number=issue_number, article_count=len(entries))

    def build_and_save(
        self,
        user: UserProfile,
        collected: list[CollectedArticle],
        issue_number: int | None = None,
        out_root: Path | None = None,
    ) -> Path:
        t0 = datetime.now().astimezone()
        self._emit_stage(user, issue_number, stage_id="save", status="start", in_count=len(collected), out_count=0)
        html = self.compose_html(user=user, collected=collected, issue_number=issue_number)

        out_root = Path(out_root or (self.log_dir / user.name.replace(" ", "_") / "outputs"))
        ensure_dir(out_root)
        date = datetime.now().strftime("%Y%m%d")
        out = out_root / f"{date}_{user.user_code}.html"
        out.write_text(html, encoding="utf-8")

        # issue no history
        (out_root / "issue_no.txt").write_text(str(int(issue_number or 0)), encoding="utf-8")

        # checksum + metadata
        digest = hashlib.sha1(html.encode("utf-8")).hexdigest()
        meta = {
            "user_code": user.user_code,
            "name": user.name,
            "issue_no": issue_number,
            "issue_date": datetime.now().strftime("%Y.%m.%d"),
            "count": len(collected),
            "sha1": digest,
            "trace_id": self._build_trace_id(user, issue_number, len(collected)),
        }
        (out_root / "meta.json").write_text(json_export(meta), encoding="utf-8")
        if self.trace_enabled:
            self._emit_stage(user, issue_number, stage_id="save", status="done", in_count=len(collected), out_count=1, elapsed_ms=int((datetime.now().astimezone()-t0).total_seconds()*1000), extra={"html": str(out)})
        return out


def json_export(obj):
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)
